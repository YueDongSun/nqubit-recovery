"""Deterministic 4Q anchor probe allocation and ephemeral sidecar lifecycle."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from system_control.experiment_control.data_protocol import atomic_write_json, sha256_file
from system_control.experiment_control.operation_points import (
    operation_index_components,
    operation_registry_hash,
)

from .contracts import OperationPhaseProbeV1
from .phase_geometry import deterministic_probe_directions


PROBE_REPEAT_DESIGN: Dict[int, tuple[float, str, int]] = {
    1: (0.00, "zero_1", 0),
    2: (0.02, "pair_a", 1),
    3: (0.02, "pair_a", -1),
    4: (0.02, "pair_b", 1),
    5: (0.02, "pair_b", -1),
    6: (0.05, "pair_c", 1),
    7: (0.05, "pair_c", -1),
    8: (0.00, "zero_2", 0),
    9: (0.05, "pair_d", 1),
    10: (0.05, "pair_d", -1),
    11: (0.08, "pair_e", 1),
    12: (0.08, "pair_e", -1),
    13: (0.08, "pair_f", 1),
    14: (0.08, "pair_f", -1),
    15: (0.00, "zero_3", 0),
}


def phase_probe_schedule_fields(plan_id: str, anchor_id: str, repeat_index: int) -> Dict[str, Any]:
    repeat = int(repeat_index)
    if repeat not in PROBE_REPEAT_DESIGN:
        raise ValueError("anchor repeat_index must be in 1..15")
    amplitude, pair_name, sign = PROBE_REPEAT_DESIGN[repeat]
    pair_id = f"{anchor_id}_{pair_name}"
    seed = _probe_seed(plan_id, anchor_id, pair_id)
    return {
        "schema": "OperationPhaseProbeScheduleV1",
        "amplitude_rad": amplitude,
        "code_pair_id": pair_id,
        "pair_sign": sign,
        "direction_seed": seed,
        "gauge_convention": "per_operation_zero_mean_helmert",
        "ephemeral": True,
        "persistent_delta_allowed": False,
    }


def build_phase_probe_tensor(
    schedule_state: Mapping[str, Any],
    operation_registry: Sequence[Mapping[str, Any] | Any],
) -> np.ndarray:
    if schedule_state.get("state_role") != "anchor":
        raise ValueError("phase probes are forbidden on coverage states")
    probe = dict(schedule_state.get("phase_probe") or {})
    if probe.get("schema") != "OperationPhaseProbeScheduleV1":
        raise ValueError("anchor is missing the V4 phase_probe schedule record")
    amplitude = float(probe["amplitude_rad"])
    sign = int(probe["pair_sign"])
    values = deterministic_probe_directions(
        operation_count=256,
        cell_count=4,
        seed=int(probe["direction_seed"]),
        amplitude_rad=amplitude,
    )
    if sign == 0:
        values[:] = 0.0
    else:
        values *= sign
    if len(operation_registry) != 256:
        raise ValueError("4Q phase probes require a 256-entry operation registry")
    return values


def phase_probe_tensor_to_cycle_matrices(phase_tensor: np.ndarray) -> list[np.ndarray]:
    """Map the operation-local [256, 4] tensor onto four W_CH1 16x16 matrices."""

    values = np.asarray(phase_tensor, dtype=float)
    if values.shape != (256, 4):
        raise ValueError("4Q phase probe tensor must have shape [256, 4]")
    if not np.all(np.isfinite(values)):
        raise ValueError("phase probe tensor must contain only finite values")
    if not np.allclose(values.mean(axis=1), 0.0, atol=1e-8):
        raise ValueError("phase probe tensor must be zero-mean within every operation")
    if float(np.max(np.abs(values), initial=0.0)) > 0.08000001:
        raise ValueError("phase probe tensor exceeds the 0.08 rad campaign limit")

    matrices = [np.zeros((16, 16), dtype=float) for _ in range(4)]
    assigned = [np.zeros((16, 16), dtype=bool) for _ in range(4)]
    for operation_index in range(256):
        cycle, _local, row, _group, columns = operation_index_components(
            "4qubit", operation_index
        )
        matrices[cycle][row, columns] = values[operation_index]
        assigned[cycle][row, columns] = True
    if not all(bool(mask.all()) for mask in assigned):
        raise AssertionError("operation registry did not cover every 4Q W_CH1 matrix element")
    return matrices


def pair_safe_split_key(schedule_state: Mapping[str, Any]) -> str:
    if schedule_state.get("state_role") != "anchor":
        return f"coverage:{schedule_state.get('coverage_index', schedule_state.get('schedule_index'))}"
    probe = dict(schedule_state.get("phase_probe") or {})
    return f"anchor:{schedule_state.get('anchor_id')}:{probe.get('code_pair_id')}"


class EphemeralPhaseProbeStore:
    """Hash-bound sidecar store with an explicit active pointer and crash clear."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.sidecar_root = self.root / "phase_probes"
        self.active_pointer = self.root / "active_phase_probe.json"

    def stage(
        self,
        *,
        plan_id: str,
        schedule_state: Mapping[str, Any],
        operation_registry: Sequence[Mapping[str, Any] | Any],
        requested_awg_matrix_hash: str,
        actual_awg_matrix_hash: str,
        temperature_context: Mapping[str, Any] | None = None,
    ) -> OperationPhaseProbeV1:
        if self.active_pointer.exists():
            raise RuntimeError("an active phase probe must be cleared before staging another probe")
        tensor = build_phase_probe_tensor(schedule_state, operation_registry)
        probe = dict(schedule_state["phase_probe"])
        anchor_id = str(schedule_state["anchor_id"])
        repeat = int(schedule_state["anchor_repeat_index"])
        probe_base_id = f"{plan_id}_{anchor_id}_{repeat:02d}_{probe['code_pair_id']}_{probe['pair_sign']:+d}"
        retry_index = 1
        while True:
            probe_id = (
                probe_base_id
                if retry_index == 1
                else f"{probe_base_id}_retry_{retry_index:02d}"
            )
            directory = self.sidecar_root / probe_id
            try:
                directory.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                # Preserve the previous crash/interruption evidence and create
                # a fresh immutable sidecar for this retry.
                retry_index += 1
        tensor_path = directory / "phase_tensor.npy"
        np.save(tensor_path, tensor, allow_pickle=False)
        now = time.time()
        contract = OperationPhaseProbeV1(
            system_type="4qubit",
            operation_registry_hash=operation_registry_hash(operation_registry),
            plan_id=plan_id,
            anchor_id=anchor_id,
            repeat_index=repeat,
            code_pair_id=str(probe["code_pair_id"]),
            pair_sign=int(probe["pair_sign"]),
            amplitude_rad=float(probe["amplitude_rad"]),
            direction_seed=int(probe["direction_seed"]),
            phase_tensor_path=str(tensor_path.relative_to(self.root)).replace("\\", "/"),
            phase_tensor_shape=list(tensor.shape),
            phase_tensor_sha256=sha256_file(tensor_path),
            requested_awg_matrix_hash=str(requested_awg_matrix_hash),
            actual_awg_matrix_hash=str(actual_awg_matrix_hash),
            started_at=now,
            ended_at=now,
            temperature_context=dict(temperature_context or {}),
        )
        manifest_path = directory / "probe_manifest.json"
        atomic_write_json(manifest_path, contract.to_json_dict())
        pointer = {
            "schema": "ActiveOperationPhaseProbeV1",
            "probe_id": probe_id,
            "probe_base_id": probe_base_id,
            "retry_index": retry_index,
            "manifest_path": str(manifest_path.relative_to(self.root)).replace("\\", "/"),
            "phase_tensor_sha256": contract.phase_tensor_sha256,
            "persistent_delta_allowed": False,
            "lifecycle_status": "staged",
            "updated_at": time.time(),
        }
        atomic_write_json(self.active_pointer, pointer)
        return contract

    def active_record(self) -> Dict[str, Any] | None:
        if not self.active_pointer.exists():
            return None
        return json.loads(self.active_pointer.read_text(encoding="utf-8"))

    def load_active_contract(self) -> OperationPhaseProbeV1:
        pointer = self.active_record()
        if pointer is None:
            raise RuntimeError("there is no active phase probe")
        manifest_path = self.root / str(pointer["manifest_path"])
        contract = OperationPhaseProbeV1.from_json_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        contract.validate_sidecar(self.root)
        return contract

    def load_active_tensor(self) -> np.ndarray:
        contract = self.load_active_contract()
        return np.load(self.root / contract.phase_tensor_path, allow_pickle=False)

    def mark_lifecycle(self, status: str, **evidence: Any) -> Dict[str, Any]:
        allowed = {"staged", "generated", "uploaded", "collecting", "saved"}
        if status not in allowed:
            raise ValueError(f"unsupported phase-probe lifecycle status: {status}")
        pointer = self.active_record()
        if pointer is None:
            raise RuntimeError("there is no active phase probe")
        current = str(pointer.get("lifecycle_status") or "")
        transitions = {
            "staged": {"generated"},
            "generated": {"uploaded"},
            "uploaded": {"collecting"},
            "collecting": {"saved"},
            "saved": set(),
        }
        if status != current and status not in transitions.get(current, set()):
            raise RuntimeError(
                f"invalid phase-probe lifecycle transition: {current or 'missing'} -> {status}"
            )
        pointer["lifecycle_status"] = status
        pointer["updated_at"] = time.time()
        pointer.update({str(key): value for key, value in evidence.items()})
        atomic_write_json(self.active_pointer, pointer)
        return pointer

    def clear(self, *, require_saved: bool = False) -> None:
        if self.active_pointer.exists():
            if require_saved:
                pointer = self.active_record() or {}
                if pointer.get("lifecycle_status") != "saved":
                    raise RuntimeError("an unsaved active phase probe cannot be cleared as completed")
            self.active_pointer.unlink()

    def recover_and_verify_cleared(self) -> bool:
        self.clear()
        return not self.active_pointer.exists()


def _probe_seed(plan_id: str, anchor_id: str, pair_id: str) -> int:
    payload = f"{plan_id}|{anchor_id}|{pair_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
