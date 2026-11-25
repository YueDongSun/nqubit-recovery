"""V9-R2 authoritative operation registry (charter section 5).

The prototype's hand-written ``sp00...``/``cycle0_op00...`` placeholder mapping is not a
hardware authority. This registry derives the authoritative operation -> W-cell mapping,
command identity and matrix hashes from ``operation_points.build_operation_registry``
(itself built from the basis config, AWG generation order, command tables and current
matrix files). Any action proposal must be bound to registry hash + command table hash +
base matrix hash + target hash + model hash; ``verify_proposal`` fails closed on any
mismatch. This module lives in system_control (not the model package): it reads real
command matrix files, and the V9 model package must stay free of hardware imports.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ml_recon.physics_hybrid_v9.contracts import V9OperationPointActionProposalV1
from system_control.experiment_control.operation_points import (
    CHANNELS,
    OperationPointSpec,
    build_operation_registry,
    operation_registry_hash,
)
from system_control.qubit_recon_files.system_specs import get_spec

# 4Q channels whose command matrices are cycle-specific (matches the generator).
_CYCLE_SPECIFIC_4Q_CHANNELS = frozenset(("W_CH1", "R_CH1", "R_CH2"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stable_payload_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=True,
                         separators=(",", ":"), default=str).encode("utf-8")
    return _sha256_bytes(payload)


def _matrix_sha256(matrix: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(matrix).tobytes())


def _command_file_names(
    system_type: str, channel: str, kind: str, n_cycles: int,
) -> Dict[int, str]:
    """Return {cycle: file_name} for a channel's command matrices.

    3Q shares a single file; 4Q W_CH1/R_CH1/R_CH2 are cycle-specific (0-based cycle
    keys -> ``{cycle + 1}`` file names, exactly as the generator).
    """
    if system_type == "4qubit" and channel in _CYCLE_SPECIFIC_4Q_CHANNELS:
        return {c: f"{channel}_{kind}_{c + 1}_4qubit.txt" for c in range(n_cycles)}
    suffix = "" if system_type == "3qubit" else "_4qubit"
    return {0: f"{channel}_{kind}{suffix}.txt"}


@dataclass(frozen=True)
class V9R2MatrixProvenanceV1:
    """Provenance of one authoritative command matrix file (or its absence)."""

    channel: str
    kind: str                 # "amp" | "phase"
    cycle: Optional[int]      # None for shared 3Q/4Q files; 0-based cycle otherwise
    file_name: str
    file_sha256: str          # "" when the file is missing
    matrix_sha256: str        # "" when missing
    shape: Tuple[int, int]
    missing: bool

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["shape"] = list(self.shape)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "V9R2MatrixProvenanceV1":
        d = dict(d)
        d["shape"] = tuple(d["shape"])
        return cls(**d)


def _load_command_matrix(root: Path, file_name: str) -> Tuple[Optional[np.ndarray], bool]:
    """Load a command matrix; returns (matrix, missing). Mirrors the generator."""
    candidate = root / file_name
    try:
        value = np.loadtxt(candidate, dtype=float)
        if value.ndim == 0:
            value = np.asarray([[float(value)]])
        return np.asarray(value, dtype=float), False
    except (OSError, ValueError):
        return None, True


@dataclass(frozen=True)
class V9R2OperationSpecV2:
    """Per-operation registry record: OperationPointSpec identity + hash bindings."""

    system_type: str
    global_index: int
    cycle_index: int
    local_index: int
    row: int
    group: int
    cell_indices: Tuple[int, ...]
    cell_coordinates: Tuple[Tuple[int, int], ...]
    frequencies_mhz: Tuple[float, ...]
    basis_label: str
    basis_amplitudes: Tuple[float, ...]
    basis_phases_rad: Tuple[float, ...]
    w_ch1_command_amplitudes: Tuple[Optional[float], ...]
    w_ch1_command_phases_rad: Tuple[Optional[float], ...]
    w_ch1_command_hash: str
    command_table_hash: str
    all_channel_command_context: Dict[str, Any]
    matrix_provenance: Dict[str, V9R2MatrixProvenanceV1]

    def command_table_payload(self) -> Dict[str, Any]:
        return {"global_index": self.global_index, **self.all_channel_command_context}

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["cell_indices"] = list(self.cell_indices)
        d["cell_coordinates"] = [list(c) for c in self.cell_coordinates]
        d["frequencies_mhz"] = list(self.frequencies_mhz)
        d["basis_amplitudes"] = list(self.basis_amplitudes)
        d["basis_phases_rad"] = list(self.basis_phases_rad)
        d["w_ch1_command_amplitudes"] = list(self.w_ch1_command_amplitudes)
        d["w_ch1_command_phases_rad"] = list(self.w_ch1_command_phases_rad)
        d["matrix_provenance"] = {k: v.to_dict() for k, v in self.matrix_provenance.items()}
        return d


@dataclass(frozen=True)
class V9R2ActionBindingV1:
    """Binds one action proposal to its full hardware identity (charter section 5).

    All five identities must agree before any materialization: registry hash, command
    table hash, base matrix hash, target hash, model checkpoint hash.
    """

    system_type: str
    operation_registry_hash: str
    command_table_hash: str
    base_matrix_hash: str
    target_hash: str
    model_checkpoint_hash: str
    binding_sha256: str

    def verify(self, *, operation_registry_hash: str, command_table_hash: str,
               base_matrix_hash: str, target_hash: str, model_checkpoint_hash: str) -> None:
        if operation_registry_hash != self.operation_registry_hash:
            raise ValueError("operation registry hash mismatch")
        if command_table_hash != self.command_table_hash:
            raise ValueError("command table hash mismatch")
        if base_matrix_hash != self.base_matrix_hash:
            raise ValueError("base matrix hash mismatch")
        if target_hash != self.target_hash:
            raise ValueError("target hash mismatch")
        if model_checkpoint_hash != self.model_checkpoint_hash:
            raise ValueError("model checkpoint hash mismatch")

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "V9R2ActionBindingV1":
        return cls(**d)


class V9R2OperationRegistryV2:
    """Authoritative, hash-bound operation registry for one system.

    Built from ``operation_points.build_operation_registry`` plus the real command
    matrix files. Provides the all-cells ``matrix_index_map`` (no identity
    placeholder), per-cycle W_CH1 base matrices, and fail-closed action binding.
    """

    def __init__(
        self,
        system_type: str,
        *,
        matrix_dir: str | Path | None = None,
        basis_config: Optional[Mapping[str, Any]] = None,
        include_ml_delta: bool = False,
    ):
        if system_type not in ("3qubit", "4qubit"):
            raise ValueError(f"system_type must be 3qubit or 4qubit, got {system_type!r}")
        self.system_type = system_type
        self.matrix_dir = Path(matrix_dir) if matrix_dir else (
            Path(__file__).resolve().parents[1] / "qubit_recon_files" / "matrix")
        self.spec = get_spec(system_type)
        self.n_operations = int(self.spec["total_operations"])
        self.n_cycles = int(self.spec.get("total_cycles", 1))
        self.cells_per_operation = int(self.spec["group_size"])
        self._base_specs = build_operation_registry(
            system_type, matrix_dir=self.matrix_dir, basis_config=basis_config,
            include_ml_delta=include_ml_delta)

        # Command matrix files referenced by this system (all channels x amp/phase).
        self.matrix_provenance: Dict[str, V9R2MatrixProvenanceV1] = {}
        for channel in CHANNELS:
            for kind in ("amp", "phase"):
                for cycle, file_name in _command_file_names(
                        system_type, channel, kind, self.n_cycles).items():
                    matrix, missing = _load_command_matrix(self.matrix_dir, file_name)
                    prov = V9R2MatrixProvenanceV1(
                        channel=channel, kind=kind,
                        cycle=None if (system_type == "3qubit" or
                                       channel not in _CYCLE_SPECIFIC_4Q_CHANNELS) else cycle,
                        file_name=file_name,
                        file_sha256=_sha256_bytes((self.matrix_dir / file_name).read_bytes())
                        if not missing else "",
                        matrix_sha256=_matrix_sha256(matrix) if not missing else "",
                        shape=tuple(matrix.shape) if not missing else (0, 0),
                        missing=missing,
                    )
                    self.matrix_provenance[f"{channel}_{kind}_{cycle}"] = prov

        self.operations: List[V9R2OperationSpecV2] = [
            self._build_operation_spec(s) for s in self._base_specs]
        self.registry_hash = operation_registry_hash(self._base_specs)
        self.command_table_hash = _stable_payload_hash(
            [op.command_table_payload() for op in self.operations])
        # Snapshot the W_CH1 base matrices at construction: the registry is an immutable
        # authority. Lazy disk reads would silently rebind to files edited afterwards.
        self._w_ch1_base: Dict[str, Dict[int, np.ndarray]] = self._load_w_ch1_bases()
        self.w_ch1_base_matrix_hash = self._hash_w_ch1_bases(self._w_ch1_base)

    # ------------------------------------------------------------------
    # per-operation derivation
    # ------------------------------------------------------------------

    def _build_operation_spec(self, s: OperationPointSpec) -> V9R2OperationSpecV2:
        w_ch1 = s.channel_commands.get("W_CH1", {})
        amp_vals = list(w_ch1.get("amplitudes", []))
        phase_vals = list(w_ch1.get("phases_rad", []))
        w_ch1_hash = _stable_payload_hash({"amplitudes": amp_vals, "phases_rad": phase_vals})
        command_payload: Dict[str, Any] = {}
        for channel in CHANNELS:
            command_payload[channel] = s.channel_commands.get(channel, {})
        command_hash = _stable_payload_hash(command_payload)
        provenance = {
            key: prov for key, prov in self.matrix_provenance.items()
            if prov.channel == "W_CH1"
            and (self.system_type != "4qubit" or prov.cycle is None or prov.cycle == s.cycle_index)
        }
        return V9R2OperationSpecV2(
            system_type=self.system_type,
            global_index=s.global_index,
            cycle_index=s.cycle_index,
            local_index=s.local_index,
            row=s.row,
            group=s.group,
            cell_indices=tuple(s.cell_indices),
            cell_coordinates=tuple(tuple(c) for c in s.cell_coordinates),
            frequencies_mhz=tuple(s.frequencies_mhz),
            basis_label=s.basis_label,
            basis_amplitudes=tuple(s.basis_amplitudes),
            basis_phases_rad=tuple(s.basis_phases_rad),
            w_ch1_command_amplitudes=tuple(amp_vals),
            w_ch1_command_phases_rad=tuple(phase_vals),
            w_ch1_command_hash=w_ch1_hash,
            command_table_hash=command_hash,
            all_channel_command_context=command_payload,
            matrix_provenance=provenance,
        )

    # ------------------------------------------------------------------
    # authoritative mapping / bases
    # ------------------------------------------------------------------

    @property
    def matrix_index_map(self) -> Dict[int, List[Tuple[int, int]]]:
        """All N W_CH1 cells per operation (no identity placeholder)."""
        return {op.global_index: [tuple(c) for c in op.cell_coordinates]
                for op in self.operations}

    def _load_w_ch1_bases(self) -> Dict[str, Dict[int, np.ndarray]]:
        """The W_CH1 command matrices the materializer starts from.

        Returns ``{amp: {cycle: matrix}, phase: {cycle: matrix}}``. 3Q has a single
        cycle 0 entry; 4Q has one matrix per cycle (cycle-specific files).
        """
        amp: Dict[int, np.ndarray] = {}
        phase: Dict[int, np.ndarray] = {}
        for key, prov in self.matrix_provenance.items():
            if prov.channel != "W_CH1" or prov.missing:
                continue
            cycle = 0 if prov.cycle is None else prov.cycle
            matrix, _ = _load_command_matrix(self.matrix_dir, prov.file_name)
            (amp if prov.kind == "amp" else phase)[cycle] = matrix
        return {"amp": amp, "phase": phase}

    @staticmethod
    def _hash_w_ch1_bases(bases: Dict[str, Dict[int, np.ndarray]]) -> str:
        """Stable hash over all W_CH1 base matrices (amp+phase, all cycles)."""
        payload = b""
        for kind in ("amp", "phase"):
            for cycle in sorted(bases[kind]):
                payload += _matrix_sha256(bases[kind][cycle]).encode("ascii")
        return _sha256_bytes(payload)

    def w_ch1_base_matrices(self) -> Dict[str, Dict[int, np.ndarray]]:
        """The construction-time snapshot of the W_CH1 base matrices."""
        return self._w_ch1_base

    @property
    def has_complete_w_ch1_base(self) -> bool:
        bases = self._w_ch1_base
        return len(bases["amp"]) == self.n_cycles and len(bases["phase"]) == self.n_cycles

    # ------------------------------------------------------------------
    # action binding (fail-closed)
    # ------------------------------------------------------------------

    def bind_action(self, *, target_hash: str,
                    model_checkpoint_hash: str) -> V9R2ActionBindingV1:
        if not self.has_complete_w_ch1_base:
            raise ValueError(
                f"cannot bind action: {self.system_type} W_CH1 base matrices incomplete "
                f"(missing command matrix files in {self.matrix_dir})")
        binding = V9R2ActionBindingV1(
            system_type=self.system_type,
            operation_registry_hash=self.registry_hash,
            command_table_hash=self.command_table_hash,
            base_matrix_hash=self.w_ch1_base_matrix_hash,
            target_hash=target_hash,
            model_checkpoint_hash=model_checkpoint_hash,
            binding_sha256="",
        )
        payload = _stable_payload_hash(binding.to_dict())
        return dataclasses.replace(binding, binding_sha256=payload)

    def bind_proposal(self, proposal: V9OperationPointActionProposalV1) -> V9OperationPointActionProposalV1:
        """Return a copy of ``proposal`` with ``control_hash`` bound to the base-matrix
        hash (the model may only fill the other four hashes)."""
        if proposal.system_type != self.system_type:
            raise ValueError(f"proposal system {proposal.system_type} != registry {self.system_type}")
        return dataclasses.replace(proposal, control_hash=self.w_ch1_base_matrix_hash)

    def verify_proposal(
        self,
        proposal: V9OperationPointActionProposalV1,
        *,
        model_checkpoint_hash: str,
    ) -> V9R2ActionBindingV1:
        """Fail-closed: every bound hash must match the authoritative registry.

        ``model_checkpoint_hash`` is the AUTHORITATIVE frozen-checkpoint hash (from the
        operator ARM contract), not the proposal's own claim — a proposal cannot certify
        itself. The proposal's registry / command-table / base-matrix hashes must match
        the registry, and its model hash must match the authoritative checkpoint.
        """
        binding = self.bind_action(
            target_hash=proposal.target_hash,
            model_checkpoint_hash=model_checkpoint_hash)
        binding.verify(
            operation_registry_hash=proposal.operation_registry_hash,
            command_table_hash=proposal.command_table_hash,
            base_matrix_hash=proposal.control_hash,
            target_hash=proposal.target_hash,
            model_checkpoint_hash=proposal.model_checkpoint_hash)
        return binding
