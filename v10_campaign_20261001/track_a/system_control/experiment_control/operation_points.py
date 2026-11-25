"""Versioned operation-point registry for the 3Q and 4Q AOD experiments.

The registry is intentionally derived from the same matrix layout used by the
sequence generators.  It is a *read-only description* of an acquired round:
building it never writes matrices, deltas, or waveforms.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

from system_control.qubit_recon_files.basis_configs import build_3qubit_basis_config, build_4qubit_basis_config
from system_control.qubit_recon_files.digital_delay_defaults import DEFAULT_DIGITAL_DELAY_OFFSET_NS, digital_delay_values
from system_control.qubit_recon_files.system_specs import get_spec
from system_control.qubit_recon_files.wch1_excitation_factor import build_wch1_excitation_factor_matrix


OPERATION_POINT_SCHEMA_VERSION = 2
CHANNELS = ("W_CH1", "W_CH2", "W_CH3", "W_CH4", "R_CH1", "R_CH2", "R_CH3", "R_CH4")
X_CHANNELS = frozenset(("W_CH1", "W_CH3", "R_CH1", "R_CH3"))


@dataclass(frozen=True)
class OperationPointSpec:
    schema_version: int
    system_type: str
    global_index: int
    cycle_index: int
    local_index: int
    row: int
    group: int
    cell_indices: List[int]
    cell_coordinates: List[tuple[int, int]]
    frequencies_mhz: List[float]
    basis_label: str
    basis_amplitudes: List[float]
    basis_phases_rad: List[float]
    channel_commands: Dict[str, Dict[str, Any]]
    diffraction_order_signs: Dict[str, float]
    operation_duration_ns: int

    def to_json_dict(self) -> Dict[str, Any]:
        """JSON-safe representation; tuples are intentionally emitted as lists."""
        return asdict(self)


def operation_registry_hash(registry: Iterable[OperationPointSpec | Mapping[str, Any]]) -> str:
    """Return a stable SHA-256 hash suitable for associating an acquisition."""
    values = [item.to_json_dict() if isinstance(item, OperationPointSpec) else dict(item) for item in registry]
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_operation_registry(
    system_type: str,
    *,
    matrix_dir: str | Path | None = None,
    basis_config: Optional[Mapping[str, Any]] = None,
    include_ml_delta: bool = True,
    diffraction_order_signs: Optional[Mapping[str, float]] = None,
    digital_timing: Optional[Mapping[str, Any]] = None,
) -> List[OperationPointSpec]:
    """Build the exact row/group/cycle ordering consumed by the AWG generators.

    ``matrix_dir`` defaults to the production matrix directory.  Missing command
    matrices are represented explicitly in a command's ``missing`` field rather
    than being silently made up; normal production trees contain all matrices.
    ``include_ml_delta`` mirrors sequence-generator overlays without writing one.
    """
    spec = get_spec(system_type)
    if basis_config is None:
        basis_config = build_3qubit_basis_config() if system_type == "3qubit" else build_4qubit_basis_config()
    matrix_root = Path(matrix_dir) if matrix_dir else Path(__file__).resolve().parents[1] / "qubit_recon_files" / "matrix"
    basis_entries = _flatten_basis_entries(system_type, basis_config)
    expected = int(spec["total_operations"])
    if len(basis_entries) != expected:
        raise ValueError(f"{system_type} basis config has {len(basis_entries)} entries, expected {expected}")

    signs = {channel: 1.0 for channel in CHANNELS}
    # The read-AOD projection convention is intentionally only on read channels.
    signs.update({"R_CH1": -1.0, "R_CH2": -1.0, "R_CH3": -1.0, "R_CH4": -1.0})
    if diffraction_order_signs:
        signs.update({str(key): float(value) for key, value in diffraction_order_signs.items()})
    timing = dict(digital_timing or _default_digital_timing())

    registry: List[OperationPointSpec] = []
    for global_index, basis in enumerate(basis_entries):
        cycle, local, row, group, cols = operation_index_components(system_type, global_index)
        commands = _channel_commands(
            system_type, matrix_root, cycle, row, group, cols, include_ml_delta, timing,
        )
        freqs = [float(spec["freq_start_mhz"] + col * spec["freq_step_mhz"]) for col in cols]
        coordinates = [(row, col) for col in cols]
        cell_indices = [row * int(spec["freq_steps"]) + col for col in cols]
        registry.append(OperationPointSpec(
            schema_version=OPERATION_POINT_SCHEMA_VERSION,
            system_type=system_type,
            global_index=global_index,
            cycle_index=cycle,
            local_index=local,
            row=row,
            group=group,
            cell_indices=cell_indices,
            cell_coordinates=coordinates,
            frequencies_mhz=freqs,
            basis_label=str(basis.get("label", f"operation_{global_index}")),
            basis_amplitudes=[float(value) for value in basis.get("amp", [])],
            basis_phases_rad=[float(value) for value in basis.get("phase", [])],
            channel_commands=commands,
            diffraction_order_signs=signs,
            operation_duration_ns=int(spec["op_duration_ns"]),
        ))
    return registry


def build_3qubit_operation_registry(**kwargs: Any) -> List[OperationPointSpec]:
    return build_operation_registry("3qubit", **kwargs)


def build_4qubit_operation_registry(**kwargs: Any) -> List[OperationPointSpec]:
    return build_operation_registry("4qubit", **kwargs)


def operation_index_components(system_type: str, global_index: int) -> tuple[int, int, int, int, List[int]]:
    """Return ``cycle, local, row, group, columns`` using generator indexing."""
    spec = get_spec(system_type)
    total = int(spec["total_operations"])
    if not 0 <= int(global_index) < total:
        raise IndexError(f"{system_type} operation index must be in [0, {total}), got {global_index}")
    if system_type == "4qubit":
        cycle = int(global_index) // int(spec["ops_per_cycle"])
        local = int(global_index) % int(spec["ops_per_cycle"])
    else:
        cycle, local = 0, int(global_index)
    row = local // int(spec["groups_per_row"])
    group = local % int(spec["groups_per_row"])
    start = group * int(spec["group_size"])
    return cycle, local, row, group, list(range(start, start + int(spec["group_size"])))


def _flatten_basis_entries(system_type: str, config: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    if config.get("system_type") not in (None, system_type):
        raise ValueError(f"basis config belongs to {config.get('system_type')}, not {system_type}")
    if system_type == "3qubit":
        return list(config.get("entries", []))
    entries: List[Mapping[str, Any]] = []
    for cycle in config.get("cycles", []):
        entries.extend(cycle.get("entries", []))
    return entries


def _channel_commands(
    system_type: str,
    matrix_root: Path,
    cycle: int,
    row: int,
    group: int,
    cols: List[int],
    include_ml_delta: bool,
    timing: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    commands: Dict[str, Dict[str, Any]] = {}
    suffix = "" if system_type == "3qubit" else "_4qubit"
    for channel in CHANNELS:
        is_x = channel in X_CHANNELS
        amp, amp_missing = _load_command_matrix(matrix_root, channel, "amp", suffix, cycle, system_type, is_x)
        phase, phase_missing = _load_command_matrix(matrix_root, channel, "phase", suffix, cycle, system_type, is_x)
        if include_ml_delta:
            amp = _overlay_delta(amp, matrix_root / "ml_delta" / _delta_name(channel, "amp", cycle, system_type))
            phase = _overlay_delta(phase, matrix_root / "ml_delta" / _delta_name(channel, "phase", cycle, system_type))
        if is_x:
            values_amp = [_matrix_value(amp, row, col) for col in cols]
            values_phase = [_matrix_value(phase, row, col) for col in cols]
            tone_indices = list(cols)
        else:
            values_amp = [_matrix_value(amp, row, group)]
            values_phase = [_matrix_value(phase, row, group)]
            tone_indices = [row]
        if channel == "W_CH1" and all(value is not None for value in values_amp):
            excitation = build_wch1_excitation_factor_matrix(get_spec(system_type)["freq_steps"])
            values_amp = [float(value) * float(excitation[row, col]) for value, col in zip(values_amp, cols)]
        commands[channel] = {
            "tone_indices": tone_indices,
            "amplitudes": values_amp,
            "phases_rad": values_phase,
            "matrix_missing": bool(amp_missing or phase_missing),
            "digital_delay_ns": _channel_timing(timing, channel, "digital_delay_ns"),
            "digital_width_ns": _channel_timing(timing, channel, "digital_width_ns"),
            "analog_delay_ns": _channel_timing(timing, channel, "analog_delay_ns"),
        }
    return commands


def _load_command_matrix(
    root: Path, channel: str, kind: str, suffix: str, cycle: int, system_type: str, is_x: bool,
) -> tuple[Optional[np.ndarray], bool]:
    # 4Q W_CH1/R_CH1/R_CH2 use cycle-specific command matrices exactly as the generator.
    if system_type == "4qubit" and channel in {"W_CH1", "R_CH1", "R_CH2"}:
        candidate = root / f"{channel}_{kind}_{cycle + 1}_4qubit.txt"
    else:
        candidate = root / f"{channel}_{kind}{suffix}.txt"
    try:
        value = np.loadtxt(candidate, dtype=float)
        if value.ndim == 0:
            value = np.asarray([[float(value)]])
        return np.asarray(value, dtype=float), False
    except (OSError, ValueError):
        return None, True


def _delta_name(channel: str, kind: str, cycle: int, system_type: str) -> str:
    if system_type == "4qubit" and channel in {"W_CH1", "R_CH1"}:
        return f"{channel}_{kind}_delta_{cycle + 1}_4qubit.txt"
    suffix = "" if system_type == "3qubit" else "_4qubit"
    return f"{channel}_{kind}_delta{suffix}.txt"


def _overlay_delta(matrix: Optional[np.ndarray], path: Path) -> Optional[np.ndarray]:
    if matrix is None or not path.exists():
        return matrix
    try:
        delta = np.loadtxt(path, dtype=float)
        if np.asarray(delta).shape == matrix.shape:
            limit = 0.03 if "_amp_" in path.name else 0.20
            return matrix + np.clip(delta, -limit, limit)
    except (OSError, ValueError):
        pass
    return matrix


def _matrix_value(matrix: Optional[np.ndarray], row: int, col: int) -> Optional[float]:
    if matrix is None or row >= matrix.shape[0] or col >= matrix.shape[1]:
        return None
    return float(matrix[row, col])


def _channel_timing(timing: Mapping[str, Any], channel: str, key: str) -> Optional[int]:
    channels = timing.get("channels", timing)
    value = channels.get(channel, {}) if isinstance(channels, Mapping) else {}
    if isinstance(value, Mapping) and value.get(key) is not None:
        return int(value[key])
    return None


def _default_digital_timing() -> Dict[str, Any]:
    delays = digital_delay_values(DEFAULT_DIGITAL_DELAY_OFFSET_NS)
    analog = {
        "W_CH1": 4300, "W_CH2": 4950, "W_CH3": 1400, "W_CH4": 0,
        "R_CH1": 1800, "R_CH2": 0, "R_CH3": 700, "R_CH4": 4900,
    }
    return {"channels": {
        channel: {
            "digital_delay_ns": int(delays[channel]),
            "digital_width_ns": 120 if channel.startswith("W_") else 400,
            "analog_delay_ns": analog[channel],
        }
        for channel in CHANNELS
    }}
