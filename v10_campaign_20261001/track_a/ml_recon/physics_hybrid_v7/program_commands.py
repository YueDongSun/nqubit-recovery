"""Resolve snapshot-bound measurement-program amplitudes and loaded phases."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from ml_recon.measured_data.contracts import MeasuredSampleRecordV1


PROGRAM_COMMAND_SCHEMA = "v7_program_command_v1"


@dataclass(frozen=True)
class ProgramCommandV1:
    sample_id: str
    system_type: str
    amplitudes: np.ndarray
    phases_rad: np.ndarray
    automation_run_id: Optional[str]
    plan_path_relative: Optional[str]
    plan_sha256: Optional[str]
    amplitude_source: str
    phase_source: str
    plan_amplitude_verified: bool
    available: bool

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema": PROGRAM_COMMAND_SCHEMA,
            "sample_id": self.sample_id,
            "system_type": self.system_type,
            "amplitudes": self.amplitudes.tolist(),
            "phases_rad": self.phases_rad.tolist(),
            "automation_run_id": self.automation_run_id,
            "plan_path_relative": self.plan_path_relative,
            "plan_sha256": self.plan_sha256,
            "amplitude_source": self.amplitude_source,
            "phase_source": self.phase_source,
            "plan_amplitude_verified": self.plan_amplitude_verified,
            "available": self.available,
            "scope": "program_state_command_not_per_channel_matrix_command",
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _phase_from_state(state: Mapping[str, Any]) -> Tuple[np.ndarray, str]:
    if state.get("actual_loaded_phases_rad") is not None:
        return np.asarray(state["actual_loaded_phases_rad"], dtype=float), "snapshot_actual_loaded"
    if state.get("actual_loaded_phases_deg") is not None:
        return np.deg2rad(np.asarray(state["actual_loaded_phases_deg"], dtype=float)), "snapshot_actual_loaded"
    if state.get("phases_rad") is not None:
        return np.asarray(state["phases_rad"], dtype=float), "snapshot_program_context"
    return np.deg2rad(np.asarray(state.get("phases_deg", []), dtype=float)), "snapshot_program_context"


def _snapshot_program_state(root: Path, record: MeasuredSampleRecordV1) -> Mapping[str, Any]:
    root_resolved = root.resolve()
    snapshot_dir = (root / record.snapshot_path_relative).resolve()
    if not snapshot_dir.is_relative_to(root_resolved):
        raise ValueError("snapshot path escapes the measured data root")
    manifest_path = snapshot_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest.get("target_state_params"), Mapping):
            return dict(manifest["target_state_params"])
    metadata_path = snapshot_dir / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        feedback = metadata.get("feedback_context") or {}
        if isinstance(feedback.get("target_state_params"), Mapping):
            return dict(feedback["target_state_params"])
    return dict(record.exact_target_state_params)


def resolve_program_commands(
    records: Iterable[MeasuredSampleRecordV1],
    *,
    data_root: str | Path,
) -> Tuple[List[ProgramCommandV1], Dict[str, Any]]:
    root = Path(data_root)
    plan_cache: Dict[str, Tuple[Path, Dict[str, Mapping[str, Any]], str]] = {}
    commands: List[ProgramCommandV1] = []
    for record in records:
        snapshot_state = dict(_snapshot_program_state(root, record))
        amplitudes = np.asarray(snapshot_state.get("amplitudes", []), dtype=float).reshape(-1)
        phases_rad, phase_source = _phase_from_state(snapshot_state)
        expected_qubits = 3 if record.system_type == "3qubit" else 4
        available = amplitudes.shape == (expected_qubits,) and phases_rad.shape == amplitudes.shape
        plan_path: Optional[Path] = None
        plan_hash: Optional[str] = None
        plan_state: Optional[Mapping[str, Any]] = None
        if record.automation_run_id:
            run_id = str(record.automation_run_id)
            if run_id not in plan_cache:
                candidate = root / "automation_runs" / run_id / "automation_plan.json"
                if candidate.is_file():
                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                    states = {
                        str(state.get("name")): state
                        for state in payload.get("states", [])
                        if isinstance(state, Mapping) and state.get("name")
                    }
                    plan_cache[run_id] = (candidate, states, _sha256_file(candidate))
            if run_id in plan_cache:
                plan_path, states, plan_hash = plan_cache[run_id]
                plan_state = states.get(str(snapshot_state.get("name")))
        verified = False
        amplitude_source = "snapshot_program_context" if available else "missing"
        if plan_state is not None and plan_state.get("amplitudes") is not None:
            plan_amplitudes = np.asarray(plan_state["amplitudes"], dtype=float).reshape(-1)
            verified = bool(
                plan_amplitudes.shape == amplitudes.shape
                and np.allclose(plan_amplitudes, amplitudes, rtol=0.0, atol=1e-9)
            )
            if verified:
                amplitude_source = "automation_plan_verified"
        commands.append(ProgramCommandV1(
            sample_id=record.sample_id,
            system_type=record.system_type,
            amplitudes=amplitudes if available else np.zeros(expected_qubits, dtype=float),
            phases_rad=phases_rad if available else np.zeros(expected_qubits, dtype=float),
            automation_run_id=record.automation_run_id,
            plan_path_relative=(
                plan_path.relative_to(root).as_posix() if plan_path is not None else None
            ),
            plan_sha256=plan_hash,
            amplitude_source=amplitude_source,
            phase_source=phase_source if available else "missing",
            plan_amplitude_verified=verified,
            available=available,
        ))
    source_distribution: Dict[str, int] = {}
    for command in commands:
        source_distribution[command.amplitude_source] = (
            source_distribution.get(command.amplitude_source, 0) + 1
        )
    audit = {
        "schema": "v7_program_command_audit_v1",
        "sample_count": len(commands),
        "available_count": sum(command.available for command in commands),
        "plan_amplitude_verified_count": sum(
            command.plan_amplitude_verified for command in commands
        ),
        "amplitude_source_distribution": dict(sorted(source_distribution.items())),
        "distinct_plan_hash_count": len({
            command.plan_sha256 for command in commands if command.plan_sha256
        }),
        "per_channel_matrix_command_available": False,
        "interpretation": (
            "real program-state amplitude/phase supervision; no claim of exact per-channel matrix commands"
        ),
    }
    return commands, audit


__all__ = ["PROGRAM_COMMAND_SCHEMA", "ProgramCommandV1", "resolve_program_commands"]
