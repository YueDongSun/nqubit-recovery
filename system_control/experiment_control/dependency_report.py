"""Dependency and runtime status reporting for the experiment UI."""

from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .feedback_evidence import summarize_wch1_feedback_evidence
from .fpga_programmer import find_quartus_pgm, fpga_bitstream_path
from .model_paths import resolve_feedback_model_path


@dataclass
class DependencyItem:
    name: str
    ok: bool
    detail: str
    required: bool = True

    def line(self) -> str:
        prefix = "OK" if self.ok else ("MISSING" if self.required else "WARN")
        return f"[{prefix}] {self.name}: {self.detail}"


def collect_dependency_items(
    *,
    repo_root: Optional[Path] = None,
    model_path: Optional[str] = None,
    retrieval_map_path: Optional[str] = None,
    awg1_endpoint: str = "",
    awg2_endpoint: str = "",
    sequence_ui_open: bool = False,
    data_ui_open: bool = False,
    com_open: bool = False,
    com_port: Optional[str] = None,
    data_frame_count: Optional[int] = None,
    driver_alive: Optional[bool] = None,
    read_thread_alive: Optional[bool] = None,
    last_frame_age_s: Optional[float] = None,
    awg_upload_running: bool = False,
    fpga_programming_enabled: bool = False,
    fpga_programming_running: bool = False,
    fpga_programmer_path: Optional[str] = None,
    accumulator_running: bool = False,
    last_sequence_dir: Optional[str] = None,
    system_type: str = "3qubit",
    feedback_context: Optional[Dict[str, Any]] = None,
) -> List[DependencyItem]:
    root = repo_root or Path(__file__).resolve().parents[2]
    items: List[DependencyItem] = []
    items.append(DependencyItem("Python executable", True, sys.executable, False))
    items.append(DependencyItem("Python version", True, platform.python_version(), False))

    for module_name, required in [
        ("PyQt5", True),
        ("numpy", True),
        ("matplotlib", True),
        ("pyvisa", True),
        ("websocket", False),
        ("websockets", False),
        ("torch", False),
        ("sklearn", False),
        ("pandas", False),
        ("psutil", False),
        ("zeroconf", False),
        ("tm_devices", False),
    ]:
        items.append(_module_item(module_name, required))
    items.append(_directml_item())

    for label, relative_path, required in [
        ("3Q phase calculator", "system_control/qubit_recon_files/phase_cal.py", True),
        ("4Q phase calculator", "system_control/qubit_recon_files/phase_cal_4qubit.py", True),
        ("3Q sequence generator", "system_control/qubit_recon_files/entangle_sequence.py", True),
        ("4Q sequence generator", "system_control/qubit_recon_files/entangle_sequence_4qubit.py", True),
        ("Calibrated density reconstruction", "system_control/qubit_recon_files/calibrated_density_reconstruction.py", True),
        ("Default idler calibration CSV", "histogram_2026-06-24_22-01-37.csv", True),
        ("3Q retrieval efficiency matrix (legacy override)", "system_control/qubit_recon_files/matrix/retrieval_efficiency_3qubit.txt", False),
        ("4Q retrieval efficiency matrix (legacy override)", "system_control/qubit_recon_files/matrix/retrieval_efficiency_4qubit.txt", False),
        ("3Q AWG uploader", "AWGcontrol_port/set_front_3qubit.py", True),
        ("4Q AWG uploader", "AWGcontrol_port/set_front_4qubit.py", True),
        ("WebSocket link test", "AWGcontrol_port/test_websocket_link.py", False),
        ("PyVISA WebSocket bridge server", "AWGcontrol_port/pyvisa_websocket_server.py", False),
    ]:
        items.append(_path_item(root, label, relative_path, required))

    bitstream = fpga_bitstream_path(system_type, root)
    bitstream_ok = bitstream.exists() and bitstream.stat().st_size > 0
    items.append(
        DependencyItem(
            f"FPGA bitstream ({system_type})",
            bitstream_ok,
            str(bitstream),
            required=fpga_programming_enabled,
        )
    )
    programmer = find_quartus_pgm(fpga_programmer_path)
    programmer_detail = str(programmer) if programmer else (fpga_programmer_path or "quartus_pgm not found")
    items.append(
        DependencyItem(
            "Quartus programmer",
            programmer is not None,
            programmer_detail,
            required=fpga_programming_enabled,
        )
    )

    if model_path:
        model = resolve_feedback_model_path(system_type, model_path, root)
        detail = str(model)
        if _normalise_path_text(model_path) != _normalise_path_text(str(model)):
            detail = f"{model} (resolved from {model_path})"
        items.append(DependencyItem("Feedback model", model.exists(), detail, required=False))
    if retrieval_map_path:
        retrieval = _resolve(root, retrieval_map_path)
        items.append(DependencyItem("Retrieval map", retrieval.exists(), str(retrieval), required=False))

    items.extend(
        [
            DependencyItem("AWG1 endpoint", bool(awg1_endpoint.strip()), awg1_endpoint or "<empty>", True),
            DependencyItem("AWG2 endpoint", bool(awg2_endpoint.strip()), awg2_endpoint or "<empty>", True),
            DependencyItem("Sequence UI", sequence_ui_open, "open" if sequence_ui_open else "not open", False),
            DependencyItem("Data UI", data_ui_open, "open" if data_ui_open else "not open", False),
            DependencyItem("COM port", com_open, _com_detail(com_open, com_port), False),
            DependencyItem(
                "FPGA programming",
                not fpga_programming_running,
                "running" if fpga_programming_running else "idle",
                False,
            ),
            DependencyItem("AWG upload", not awg_upload_running, "running" if awg_upload_running else "idle", False),
            DependencyItem(
                "Measurement window",
                not accumulator_running,
                "running" if accumulator_running else "idle",
                False,
            ),
        ]
    )
    if driver_alive is not None:
        items.append(DependencyItem("Serial driver", bool(driver_alive), "alive" if driver_alive else "idle", False))
    if read_thread_alive is not None:
        items.append(
            DependencyItem(
                "Serial read thread",
                bool(read_thread_alive),
                "alive" if read_thread_alive else "not running",
                False,
            )
        )
    if data_frame_count is not None:
        items.append(DependencyItem("Data frames", True, str(int(data_frame_count)), False))
    if last_frame_age_s is not None:
        items.append(DependencyItem("Last frame age", True, f"{last_frame_age_s:.2f} s", False))
    if last_sequence_dir:
        seq_dir = _resolve(root, last_sequence_dir)
        items.append(DependencyItem("Last sequence dir", seq_dir.exists(), str(seq_dir), required=False))
    if feedback_context is not None:
        evidence = summarize_wch1_feedback_evidence(feedback_context, system_type)
        items.append(DependencyItem("W_CH1 feedback evidence", evidence.ok, evidence.detail, False))
    return items


def format_dependency_report(items: Iterable[DependencyItem]) -> str:
    return "\n".join(item.line() for item in items)


def has_blocking_missing(items: Iterable[DependencyItem]) -> bool:
    return any(item.required and not item.ok for item in items)


def _module_item(module_name: str, required: bool) -> DependencyItem:
    return DependencyItem(
        module_name,
        importlib.util.find_spec(module_name) is not None,
        "importable" if importlib.util.find_spec(module_name) is not None else "not importable",
        required,
    )


def _directml_item() -> DependencyItem:
    if importlib.util.find_spec("torch_directml") is None:
        return DependencyItem("DirectML", True, "not installed; CPU/CUDA auto path will be used", False)
    try:
        import torch_directml

        available = bool(torch_directml.is_available())
        detail = "available but only used when device=directml" if available else "installed but not available"
        return DependencyItem("DirectML", True, detail, False)
    except Exception as exc:
        return DependencyItem("DirectML", True, f"installed but probe failed: {exc}", False)


def _path_item(root: Path, label: str, relative_path: str, required: bool) -> DependencyItem:
    path = root / relative_path
    return DependencyItem(label, path.exists(), str(path), required)


def _resolve(root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return root / path


def _com_detail(opened: bool, port: Optional[str]) -> str:
    label = port or "<unknown>"
    return f"{label} open" if opened else f"{label} not open"


def _normalise_path_text(path_text: str) -> str:
    return str(Path(str(path_text).replace("\\", "/"))).replace("\\", "/").lower()
