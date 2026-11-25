"""Quartus FPGA programming helpers for the 3Q/4Q automation flow."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


VALID_SYSTEM_TYPES = {"3qubit", "4qubit"}
DEFAULT_PROJECT_NAME = "LatticeWriteRead"
ENV_PROGRAMMER = "EXPERIMENT_FPGA_PROGRAMMER"
ENV_CABLE = "EXPERIMENT_FPGA_CABLE"
ENV_TIMEOUT_S = "EXPERIMENT_FPGA_PROGRAM_TIMEOUT_S"
LOCAL_DEFAULT_PROGRAMMER_PATH = Path("D:/quartus/13.1/quartus/bin64/quartus_pgm.exe")


@dataclass(frozen=True)
class FpgaProgramTarget:
    system_type: str
    project_dir: Path
    bitstream_path: Path


@dataclass(frozen=True)
class FpgaProgramResult:
    system_type: str
    project_dir: str
    bitstream_path: str
    programmer_path: str
    command: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""

    def to_json_dict(self) -> dict:
        return {
            "system_type": self.system_type,
            "project_dir": self.project_dir,
            "bitstream_path": self.bitstream_path,
            "programmer_path": self.programmer_path,
            "command": list(self.command),
            "returncode": int(self.returncode),
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def fpga_project_dir(system_type: str, repo_root: Optional[Path] = None) -> Path:
    system_type = _normalise_system_type(system_type)
    root = repo_root or repo_root_from_here()
    return Path(root) / "VerilogHDL" / system_type


def fpga_bitstream_path(
    system_type: str,
    repo_root: Optional[Path] = None,
    project_name: str = DEFAULT_PROJECT_NAME,
) -> Path:
    return fpga_project_dir(system_type, repo_root) / "output_files" / f"{project_name}.sof"


def resolve_fpga_target(system_type: str, repo_root: Optional[Path] = None) -> FpgaProgramTarget:
    system_type = _normalise_system_type(system_type)
    project_dir = fpga_project_dir(system_type, repo_root)
    bitstream = fpga_bitstream_path(system_type, repo_root)
    if not project_dir.exists():
        raise FileNotFoundError(f"FPGA project directory not found: {project_dir}")
    if not bitstream.exists():
        raise FileNotFoundError(f"FPGA bitstream not found: {bitstream}")
    if bitstream.stat().st_size <= 0:
        raise FileNotFoundError(f"FPGA bitstream is empty: {bitstream}")
    return FpgaProgramTarget(
        system_type=system_type,
        project_dir=project_dir,
        bitstream_path=bitstream,
    )


def find_quartus_pgm(configured_path: Optional[str] = None) -> Optional[Path]:
    """Resolve quartus_pgm from an explicit path, env var, PATH, or common installs."""

    configured = _clean_path_text(configured_path)
    if configured and not _is_generic_programmer_name(configured):
        candidate = Path(configured)
        return candidate if candidate.exists() else None

    configured = _clean_path_text(os.environ.get(ENV_PROGRAMMER))
    if configured and not _is_generic_programmer_name(configured):
        candidate = Path(configured)
        return candidate if candidate.exists() else None

    if LOCAL_DEFAULT_PROGRAMMER_PATH.exists():
        return LOCAL_DEFAULT_PROGRAMMER_PATH

    for name in ("quartus_pgm", "quartus_pgm.exe"):
        found = shutil.which(name)
        if found:
            return Path(found)

    for candidate in _common_quartus_candidates():
        if candidate.exists():
            return candidate
    return None


def build_quartus_program_command(
    programmer_path: Path,
    target: FpgaProgramTarget,
    cable: Optional[str] = None,
) -> list[str]:
    command = [str(programmer_path), "-m", "JTAG"]
    cable = (cable or os.environ.get(ENV_CABLE) or "").strip()
    if cable:
        command.extend(["-c", cable])
    command.extend(["-o", f"p;{target.bitstream_path}"])
    return command


def program_fpga(
    system_type: str,
    *,
    repo_root: Optional[Path] = None,
    programmer_path: Optional[str] = None,
    cable: Optional[str] = None,
    timeout_s: Optional[float] = None,
    log: Optional[Callable[[str], None]] = None,
) -> FpgaProgramResult:
    """Program the selected 3Q/4Q FPGA .sof with quartus_pgm.

    This programs the already-built ``output_files/LatticeWriteRead.sof``.
    Rebuilding the Quartus project is intentionally not part of the automation
    path because a compile is long-running and should be an explicit operator
    action.
    """

    target = resolve_fpga_target(system_type, repo_root)
    programmer = find_quartus_pgm(programmer_path)
    if programmer is None:
        configured = programmer_path or os.environ.get(ENV_PROGRAMMER) or "quartus_pgm"
        raise FileNotFoundError(
            f"Quartus programmer not found ({configured}). "
            f"Set {ENV_PROGRAMMER} or put quartus_pgm on PATH."
        )

    command = build_quartus_program_command(programmer, target, cable)
    timeout = _timeout(timeout_s)
    _emit(log, f"FPGA target: {target.system_type} -> {target.bitstream_path}")
    _emit(log, f"Running Quartus programmer: {_command_for_log(command)}")
    completed = subprocess.run(
        command,
        cwd=str(target.project_dir),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    result = FpgaProgramResult(
        system_type=target.system_type,
        project_dir=str(target.project_dir),
        bitstream_path=str(target.bitstream_path),
        programmer_path=str(programmer),
        command=command,
        returncode=int(completed.returncode),
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )
    _emit_process_output(log, result.stdout, result.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Quartus FPGA programming failed with exit code {completed.returncode}. "
            f"Bitstream: {target.bitstream_path}"
        )
    _emit(log, f"FPGA programming complete for {target.system_type}.")
    return result


def _normalise_system_type(system_type: str) -> str:
    value = str(system_type or "").lower().replace("_", "")
    if value in {"3q", "3qubit", "3qubits"}:
        return "3qubit"
    if value in {"4q", "4qubit", "4qubits"}:
        return "4qubit"
    raise ValueError("system_type must be '3qubit' or '4qubit'")


def _clean_path_text(path_text: Optional[str]) -> str:
    return str(path_text or "").strip().strip('"')


def _is_generic_programmer_name(path_text: str) -> bool:
    return path_text.lower() in {"quartus_pgm", "quartus_pgm.exe"}


def _common_quartus_candidates() -> list[Path]:
    roots = [
        Path(os.environ[var])
        for var in ("QUARTUS_ROOTDIR", "QUARTUS_ROOTDIR_OVERRIDE")
        if os.environ.get(var)
    ]
    roots.extend(
        [
            Path("D:/quartus"),
            Path("C:/intelFPGA_lite"),
            Path("C:/intelFPGA"),
            Path("C:/altera"),
            Path("C:/Program Files/IntelFPGA_lite"),
            Path("C:/Program Files/IntelFPGA"),
        ]
    )

    candidates: list[Path] = []
    for root in roots:
        if root.name.lower() == "quartus":
            candidates.extend([root / "bin64" / "quartus_pgm.exe", root / "bin" / "quartus_pgm.exe"])
            continue
        for pattern in ("*/quartus/bin64/quartus_pgm.exe", "*/quartus/bin/quartus_pgm.exe"):
            try:
                candidates.extend(root.glob(pattern))
            except OSError:
                continue
    return candidates


def _timeout(timeout_s: Optional[float]) -> float:
    if timeout_s is not None:
        return float(timeout_s)
    value = os.environ.get(ENV_TIMEOUT_S)
    if value:
        return float(value)
    return 300.0


def _emit(log: Optional[Callable[[str], None]], message: str) -> None:
    if log is not None:
        log(message)


def _emit_process_output(log: Optional[Callable[[str], None]], stdout: str, stderr: str) -> None:
    if log is None:
        return
    for stream_name, text in (("stdout", stdout), ("stderr", stderr)):
        for line in (text or "").splitlines():
            if line.strip():
                log(f"quartus_pgm {stream_name}: {line}")


def _command_for_log(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)
