"""Deterministic, preview-first Track B reproduction through the legacy AWG generators.

This adapter is intentionally outside the research model package.  It binds a
Track B grouped action to the existing 225/256-point generators, writes only
when the caller explicitly requests it, and has no uploader or activation
dependency.  A reproduction receipt is useful evidence of byte stability; it
is not hardware or scientific validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import ast
import base64
import binascii
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from ml_recon.physics_hybrid_v10.trackb_realfd_v1.contracts import (
    canonical_json_bytes,
    canonical_sha256,
    parse_canonical_json_object,
    ResearchRFActionProposalV1,
    TrackBProposalEligibilityReceiptV1,
    require_exact_canonical_json_round_trip,
    require_git_oid,
    require_sha256,
)
from ml_recon.physics_hybrid_v10.trackb_realfd_v1.coordinate_abi import (
    RFCommandDeltaV1,
    coordinate_layout_sha,
    grouped_to_mode_action,
)
from ml_recon.physics_hybrid_v10.trackb_realfd_v1.rf_mapping_core import RegisteredRFMappingCoreV1
from ml_recon.physics_hybrid_v10.trackb_realfd_v1.rf_to_awg_mapping import (
    RFToAWGCoordinateBindingV1,
    RegisteredRFToAWGMappingV1,
    registered_rf_to_awg_recipe_sha,
)
from ml_recon.physics_hybrid_v10.trackb_realfd_v1.real_csm import array_identity_sha256


SYSTEM_GENERATOR_MODULES = {
    "3qubit": "system_control.awg225seq",
    "4qubit": "system_control.awg256seq",
}
SYSTEM_GRID_STEPS = {"3qubit": 15, "4qubit": 16}
WAVEFORM_CHANNELS = (
    "W_CH1", "W_CH2", "W_CH3", "W_CH4",
    "R_CH1", "R_CH2", "R_CH3", "R_CH4",
)
WAVEFORM_VARIANTS = ("regular", "linear", "spiral")
AWG_PHASE_CHANNELS = ("W_CH2", "W_CH4", "R_CH2", "R_CH4")
AWG_RECIPE_AXES = frozenset({"row", "column"})
BOUND_AWG_ARTIFACT_NAMES = (
    "proposal",
    "eligibility_receipt",
    "rf_delta",
    "rf_core",
    "rf_to_awg_mapping",
    "command_table_source",
    "channel_mapping_source",
    "current_control_snapshot",
)
LEGACY_AWG_CHANNEL_ANALOG_DELAYS = {
    "W_CH1": 3970,
    "W_CH2": 4800,
    "W_CH3": 1750,
    "W_CH4": 0,
    "R_CH1": 1950,
    "R_CH2": 0,
    "R_CH3": 800,
    "R_CH4": 5100,
}


@dataclass(frozen=True)
class LegacyAWGGeneratorManifestV1:
    """Static, source-hashed legacy generator configuration for safe preview."""

    system_type: str
    module_name: str
    source_relative_path: str
    source_sha256: str
    frequency_start_mhz: float
    frequency_end_mhz: float
    frequency_steps: int
    sampling_rate_hz: float
    point_duration_ns: int
    channel_analog_delays: Mapping[str, int]

    def __post_init__(self) -> None:
        if self.system_type not in SYSTEM_GENERATOR_MODULES \
                or self.module_name != SYSTEM_GENERATOR_MODULES[self.system_type]:
            raise ValueError("legacy AWG generator manifest system/module is invalid")
        require_sha256(self.source_sha256, "legacy_generator_source_sha")
        if self.frequency_steps != SYSTEM_GRID_STEPS[self.system_type] \
                or self.point_duration_ns <= 0 or self.sampling_rate_hz <= 0.0:
            raise ValueError("legacy AWG generator manifest timing/grid is invalid")
        if tuple(sorted(self.channel_analog_delays)) != tuple(sorted(WAVEFORM_CHANNELS)) \
                or any(type(value) is not int or value < 0 for value in self.channel_analog_delays.values()):
            raise ValueError("legacy AWG generator manifest analog delays are invalid")

    @property
    def source_path(self) -> Path:
        return Path(__file__).resolve().parents[3] / self.source_relative_path

    def static_config_payload(self) -> dict[str, object]:
        return {
            "schema": "TrackBAWGLegacyGeneratorStaticConfigV1",
            "system_type": self.system_type,
            "module_name": self.module_name,
            "source_relative_path": self.source_relative_path,
            "source_sha256": self.source_sha256,
            "frequency_start_mhz": self.frequency_start_mhz,
            "frequency_end_mhz": self.frequency_end_mhz,
            "frequency_steps": self.frequency_steps,
            "sampling_rate_hz": self.sampling_rate_hz,
            "point_duration_ns": self.point_duration_ns,
            "channel_analog_delays": {
                channel: int(self.channel_analog_delays[channel])
                for channel in sorted(self.channel_analog_delays)
            },
        }

    def sha256(self) -> str:
        return canonical_sha256(self.static_config_payload())


LEGACY_AWG_GENERATOR_MANIFESTS = {
    "3qubit": LegacyAWGGeneratorManifestV1(
        system_type="3qubit",
        module_name="system_control.awg225seq",
        source_relative_path="system_control/awg225seq.py",
        source_sha256="1bafbf93b32d33caaf59b8419939a890a508d8720fe290e22a8c9c133b2540b2",
        frequency_start_mhz=85.0,
        frequency_end_mhz=120.0,
        frequency_steps=15,
        sampling_rate_hz=1.0e9,
        point_duration_ns=1000,
        channel_analog_delays=LEGACY_AWG_CHANNEL_ANALOG_DELAYS,
    ),
    "4qubit": LegacyAWGGeneratorManifestV1(
        system_type="4qubit",
        module_name="system_control.awg256seq",
        source_relative_path="system_control/awg256seq.py",
        source_sha256="2066d75b009cdef5b8af2a1f893cc9dd984552e01b7a20badddec7b4607c74ba",
        frequency_start_mhz=85.0,
        frequency_end_mhz=122.5,
        frequency_steps=16,
        sampling_rate_hz=1.0e9,
        point_duration_ns=1000,
        channel_analog_delays=LEGACY_AWG_CHANNEL_ANALOG_DELAYS,
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_source_sha256(path: Path) -> str:
    """Hash UTF-8 source after normalizing Git checkout line endings."""
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("AWG generator source must be UTF-8 text") from error
    normalized = source.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _bundle_sha(root: Path, relative_paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\x00")
    return digest.hexdigest()


def _file_manifest(root: Path, relative_paths: Sequence[str]) -> tuple[tuple[str, int, str], ...]:
    return tuple(
        (relative, int((root / relative).stat().st_size), _sha256_file(root / relative))
        for relative in sorted(relative_paths)
    )


def _assert_no_transport_dependency(
    generator_manifest: LegacyAWGGeneratorManifestV1,
    generator_source_bytes: bytes,
) -> tuple[str, str]:
    """Statically reject transport imports before writing a reproduction bundle.

    The adapter purposefully invokes only the legacy waveform generator.  A
    zero counter on an unconnected spy would not prove that property, so this
    audit examines both the adapter and selected generator imports and fails if
    an AWG uploader/activation transport dependency is introduced.
    """
    forbidden_fragments = (
        "awgcontrol_port", "awg_uploader", "set_front_", "pyvisa", "websocket",
        "frontpanel", "activate_for_trial",
    )
    source_inputs = (
        ("system_control.experiment_control.v10_trackb.awg_reproduction", Path(__file__).resolve(), None),
        (generator_manifest.module_name, generator_manifest.source_path, generator_source_bytes),
    )
    identities: list[dict[str, str]] = []
    for module_name, path, supplied_bytes in source_inputs:
        if supplied_bytes is None and not path.is_file():
            raise RuntimeError("AWG reproduction audit source file is missing")
        source_bytes = path.read_bytes() if supplied_bytes is None else supplied_bytes
        try:
            source = source_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError("AWG reproduction audit source is not UTF-8") from error
        tree = ast.parse(source, filename=str(path))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        normalized = "\n".join(imported).lower()
        if any(fragment in normalized for fragment in forbidden_fragments):
            raise RuntimeError("AWG reproduction source imports a forbidden transport dependency")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function_name = ""
            if isinstance(node.func, ast.Name):
                function_name = node.func.id
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                function_name = f"{node.func.value.id}.{node.func.attr}"
            if function_name not in {"__import__", "import_module", "importlib.import_module"}:
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant) \
                    or not isinstance(node.args[0].value, str):
                raise RuntimeError("AWG reproduction source has an unanalyzable dynamic import")
            if any(fragment in node.args[0].value.lower() for fragment in forbidden_fragments):
                raise RuntimeError("AWG reproduction source dynamically imports a forbidden transport dependency")
        normalized_source = source.replace("\r\n", "\n").replace("\r", "\n")
        identities.append({
            "module": module_name,
            "sha256": hashlib.sha256(normalized_source.encode("utf-8")).hexdigest(),
        })
    if identities[-1]["sha256"] != generator_manifest.source_sha256:
        raise RuntimeError("AWG reproduction generator source differs from its registered static manifest")
    audit_sha = canonical_sha256({
        "schema": "TrackBAWGNoTransportDependencyAuditV1",
        "sources": identities,
        "generator_manifest": generator_manifest.static_config_payload(),
        "forbidden_import_fragments": list(forbidden_fragments),
    })
    return "NO_UPLOAD_DEPENDENCY_IN_INSPECTED_SOURCE", audit_sha


@dataclass(frozen=True)
class TrackBAWGCommandTableRowV1:
    """One ordered native basis-command row used by the AWG source boundary."""

    cycle_index: int
    within_cycle_index: int
    label: str
    amplitude: tuple[float, ...]
    phase_rad: tuple[float, ...]
    schema: str = field(default="TrackBAWGCommandTableRowV1", init=False)

    def __post_init__(self) -> None:
        if type(self.cycle_index) is not int or self.cycle_index < 0 \
                or type(self.within_cycle_index) is not int or self.within_cycle_index < 0:
            raise ValueError("AWG command-table row indices must be nonnegative native integers")
        if type(self.label) is not str or not self.label:
            raise ValueError("AWG command-table row label must be nonempty")
        for name in ("amplitude", "phase_rad"):
            values = tuple(getattr(self, name))
            if len(values) not in (3, 4) or any(
                type(value) is bool or not isinstance(value, (int, float))
                or not np.isfinite(float(value))
                for value in values
            ):
                raise ValueError(f"AWG command-table row {name} must be finite native 3Q/4Q values")
            object.__setattr__(self, name, tuple(float(value) for value in values))
        if len(self.amplitude) != len(self.phase_rad):
            raise ValueError("AWG command-table row amplitude/phase sizes must align")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "cycle_index": self.cycle_index,
            "within_cycle_index": self.within_cycle_index,
            "label": self.label,
            "amplitude": list(self.amplitude),
            "phase_rad": list(self.phase_rad),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TrackBAWGCommandTableRowV1":
        expected = {
            "schema", "cycle_index", "within_cycle_index", "label", "amplitude", "phase_rad",
        }
        if set(payload) != expected or payload.get("schema") != "TrackBAWGCommandTableRowV1":
            raise ValueError("AWG command-table row schema is invalid")
        amplitude = payload["amplitude"]
        phase_rad = payload["phase_rad"]
        if type(amplitude) is not list or type(phase_rad) is not list:
            raise TypeError("AWG command-table row vectors must be JSON arrays")
        return cls(
            cycle_index=payload["cycle_index"],
            within_cycle_index=payload["within_cycle_index"],
            label=payload["label"],
            amplitude=tuple(amplitude),
            phase_rad=tuple(phase_rad),
        )


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("AWG command-table source raw JSON contains duplicate keys")
        result[key] = value
    return result


def _ordered_command_table_rows_from_raw_json(
    *, system_type: str, raw_json_bytes: bytes,
) -> tuple[TrackBAWGCommandTableRowV1, ...]:
    """Parse the raw V2.4 basis table and retain its physical row order exactly."""
    try:
        decoded = json.loads(raw_json_bytes.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("AWG command-table source raw bytes must be UTF-8 JSON") from error
    if type(decoded) is not dict or decoded.get("system_type") != system_type:
        raise ValueError("AWG command-table source raw JSON system type is invalid")
    mode_count = 3 if system_type == "3qubit" else 4
    raw_rows: list[tuple[int, int, object]] = []
    if system_type == "3qubit":
        entries = decoded.get("entries")
        if type(entries) is not list:
            raise ValueError("3Q AWG command-table source must contain ordered entries")
        raw_rows = [(0, index, entry) for index, entry in enumerate(entries)]
    else:
        cycles = decoded.get("cycles")
        if type(cycles) is not list:
            raise ValueError("4Q AWG command-table source must contain ordered cycles")
        for cycle_index, cycle in enumerate(cycles):
            if type(cycle) is not dict or type(cycle.get("entries")) is not list:
                raise ValueError("4Q AWG command-table source cycle is invalid")
            raw_rows.extend(
                (cycle_index, within_cycle_index, entry)
                for within_cycle_index, entry in enumerate(cycle["entries"])
            )
    expected_rows = {"3qubit": 75, "4qubit": 256}[system_type]
    if len(raw_rows) != expected_rows:
        raise ValueError("AWG command-table source row count disagrees with the registered system")
    rows = []
    for cycle_index, within_cycle_index, entry in raw_rows:
        if type(entry) is not dict or set(entry) != {"amp", "phase", "label"}:
            raise ValueError("AWG command-table source row fields are invalid")
        amplitude = entry["amp"]
        phase_rad = entry["phase"]
        if type(amplitude) is not list or type(phase_rad) is not list:
            raise TypeError("AWG command-table source amp/phase must be JSON arrays")
        row = TrackBAWGCommandTableRowV1(
            cycle_index=cycle_index,
            within_cycle_index=within_cycle_index,
            label=entry["label"],
            amplitude=tuple(amplitude),
            phase_rad=tuple(phase_rad),
        )
        if len(row.amplitude) != mode_count:
            raise ValueError("AWG command-table source row mode count disagrees with its system")
        rows.append(row)
    return tuple(rows)


@dataclass(frozen=True)
class TrackBAWGCommandTableSourceV2:
    """Rehashable native V2.4 command table plus its complete ordered rows.

    Unlike the legacy V1 digest envelope, V2 carries the exact raw JSON bytes
    whose SHA is the command-table identity.  The consumer reparses those bytes
    into typed rows before it permits an isolated waveform write.
    """

    system_type: str
    command_table_sha: str
    frequency_steps: int
    generator_source_sha: str
    current_control_snapshot_sha: str
    raw_command_table_json_b64: str
    ordered_rows: tuple[TrackBAWGCommandTableRowV1, ...]
    schema: str = field(default="TrackBAWGCommandTableSourceV2", init=False)

    def __post_init__(self) -> None:
        if self.system_type not in SYSTEM_GRID_STEPS \
                or type(self.frequency_steps) is not int \
                or self.frequency_steps != SYSTEM_GRID_STEPS[self.system_type]:
            raise ValueError("AWG command-table source system/grid is invalid")
        for name in (
            "command_table_sha", "generator_source_sha", "current_control_snapshot_sha",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.raw_command_table_json_b64) is not str:
            raise TypeError("AWG command-table source raw JSON must be base64 text")
        try:
            raw_json_bytes = base64.b64decode(self.raw_command_table_json_b64.encode("ascii"), validate=True)
        except (UnicodeEncodeError, ValueError, binascii.Error) as error:
            raise ValueError("AWG command-table source raw JSON base64 is invalid") from error
        if hashlib.sha256(raw_json_bytes).hexdigest() != self.command_table_sha:
            raise ValueError("AWG command-table source raw bytes do not match command-table SHA")
        parsed_rows = _ordered_command_table_rows_from_raw_json(
            system_type=self.system_type,
            raw_json_bytes=raw_json_bytes,
        )
        if not self.ordered_rows or any(type(row) is not TrackBAWGCommandTableRowV1 for row in self.ordered_rows):
            raise TypeError("AWG command-table source requires exact ordered command-table rows")
        if self.ordered_rows != parsed_rows:
            raise ValueError("AWG command-table source typed rows do not match its raw command table")

    @property
    def raw_command_table_json_bytes(self) -> bytes:
        return base64.b64decode(self.raw_command_table_json_b64.encode("ascii"), validate=True)

    @classmethod
    def from_raw_command_table_json_bytes(
        cls,
        *, system_type: str, raw_command_table_json_bytes: bytes, frequency_steps: int,
        generator_source_sha: str, current_control_snapshot_sha: str,
    ) -> "TrackBAWGCommandTableSourceV2":
        return cls(
            system_type=system_type,
            command_table_sha=hashlib.sha256(raw_command_table_json_bytes).hexdigest(),
            frequency_steps=frequency_steps,
            generator_source_sha=generator_source_sha,
            current_control_snapshot_sha=current_control_snapshot_sha,
            raw_command_table_json_b64=base64.b64encode(raw_command_table_json_bytes).decode("ascii"),
            ordered_rows=_ordered_command_table_rows_from_raw_json(
                system_type=system_type,
                raw_json_bytes=raw_command_table_json_bytes,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "system_type": self.system_type,
            "command_table_sha": self.command_table_sha,
            "frequency_steps": self.frequency_steps,
            "generator_source_sha": self.generator_source_sha,
            "current_control_snapshot_sha": self.current_control_snapshot_sha,
            "raw_command_table_json_b64": self.raw_command_table_json_b64,
            "ordered_rows": [row.to_dict() for row in self.ordered_rows],
        }

    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TrackBAWGCommandTableSourceV2":
        expected = {
            "schema", "system_type", "command_table_sha", "frequency_steps",
            "generator_source_sha", "current_control_snapshot_sha", "raw_command_table_json_b64",
            "ordered_rows",
        }
        if set(payload) != expected or payload.get("schema") != "TrackBAWGCommandTableSourceV2":
            raise ValueError("AWG command-table source schema is invalid")
        rows = payload["ordered_rows"]
        if type(rows) is not list:
            raise TypeError("AWG command-table source rows must be a JSON array")
        return cls(
            system_type=payload["system_type"],
            command_table_sha=payload["command_table_sha"],
            frequency_steps=payload["frequency_steps"],
            generator_source_sha=payload["generator_source_sha"],
            current_control_snapshot_sha=payload["current_control_snapshot_sha"],
            raw_command_table_json_b64=payload["raw_command_table_json_b64"],
            ordered_rows=tuple(TrackBAWGCommandTableRowV1.from_dict(row) for row in rows),
        )

    @classmethod
    def from_canonical_json_bytes(cls, payload: bytes) -> "TrackBAWGCommandTableSourceV2":
        decoded = parse_canonical_json_object(
            payload,
            schema="TrackBAWGCommandTableSourceV2",
            exact_fields=frozenset({
                "schema", "system_type", "command_table_sha", "frequency_steps",
                "generator_source_sha", "current_control_snapshot_sha", "raw_command_table_json_b64",
                "ordered_rows",
            }),
        )
        source = cls.from_dict(decoded)
        require_exact_canonical_json_round_trip(payload, source)
        return source


@dataclass(frozen=True)
class TrackBAWGChannelMappingSourceV1:
    """Typed source envelope for all ordered RF-to-AWG coordinate bindings."""

    system_type: str
    command_table_sha: str
    frequency_steps: int
    current_control_snapshot_sha: str
    bindings: tuple[RFToAWGCoordinateBindingV1, ...]
    schema: str = field(default="TrackBAWGChannelMappingSourceV1", init=False)

    def __post_init__(self) -> None:
        if self.system_type not in SYSTEM_GRID_STEPS \
                or type(self.frequency_steps) is not int \
                or self.frequency_steps != SYSTEM_GRID_STEPS[self.system_type]:
            raise ValueError("AWG channel-mapping source system/grid is invalid")
        require_sha256(self.command_table_sha, "command_table_sha")
        require_sha256(self.current_control_snapshot_sha, "current_control_snapshot_sha")
        if not self.bindings or any(type(value) is not RFToAWGCoordinateBindingV1 for value in self.bindings):
            raise ValueError("AWG channel-mapping source requires exact coordinate bindings")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "system_type": self.system_type,
            "command_table_sha": self.command_table_sha,
            "frequency_steps": self.frequency_steps,
            "current_control_snapshot_sha": self.current_control_snapshot_sha,
            "bindings": [binding.to_dict() for binding in self.bindings],
        }

    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TrackBAWGChannelMappingSourceV1":
        expected = {
            "schema", "system_type", "command_table_sha", "frequency_steps",
            "current_control_snapshot_sha", "bindings",
        }
        if set(payload) != expected or payload.get("schema") != "TrackBAWGChannelMappingSourceV1":
            raise ValueError("AWG channel-mapping source schema is invalid")
        bindings = payload["bindings"]
        if type(bindings) is not list:
            raise TypeError("AWG channel-mapping source bindings must be a JSON array")
        return cls(
            system_type=payload["system_type"],
            command_table_sha=payload["command_table_sha"],
            frequency_steps=payload["frequency_steps"],
            current_control_snapshot_sha=payload["current_control_snapshot_sha"],
            bindings=tuple(RFToAWGCoordinateBindingV1.from_dict(value) for value in bindings),
        )

    @classmethod
    def from_canonical_json_bytes(cls, payload: bytes) -> "TrackBAWGChannelMappingSourceV1":
        decoded = parse_canonical_json_object(
            payload,
            schema="TrackBAWGChannelMappingSourceV1",
            exact_fields=frozenset({
                "schema", "system_type", "command_table_sha", "frequency_steps",
                "current_control_snapshot_sha", "bindings",
            }),
        )
        source = cls.from_dict(decoded)
        require_exact_canonical_json_round_trip(payload, source)
        return source


@dataclass(frozen=True)
class TrackBAWGCurrentControlSnapshotV1:
    """Typed current-control baseline consumed by the legacy AWG adapter."""

    system_type: str
    current_amplitude: tuple[float, ...]
    current_phase_rad: tuple[float, ...]
    schema: str = field(default="TrackBAWGCurrentControlSnapshotV1", init=False)

    def __post_init__(self) -> None:
        if self.system_type not in SYSTEM_GRID_STEPS:
            raise ValueError("AWG current-control snapshot system type is invalid")
        for name in ("current_amplitude", "current_phase_rad"):
            values = tuple(getattr(self, name))
            if not values or any(
                type(value) is bool or not isinstance(value, (int, float))
                or not np.isfinite(float(value))
                for value in values
            ):
                raise ValueError(f"{name} must be finite native numbers for the registered system")
            object.__setattr__(self, name, tuple(float(value) for value in values))
        if len(self.current_amplitude) != len(self.current_phase_rad):
            raise ValueError("AWG current-control snapshot amplitude/phase lengths must align")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "system_type": self.system_type,
            "current_amplitude": list(self.current_amplitude),
            "current_phase_rad": list(self.current_phase_rad),
        }

    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TrackBAWGCurrentControlSnapshotV1":
        expected = {"schema", "system_type", "current_amplitude", "current_phase_rad"}
        if set(payload) != expected or payload.get("schema") != "TrackBAWGCurrentControlSnapshotV1":
            raise ValueError("AWG current-control snapshot schema is invalid")
        amplitude = payload["current_amplitude"]
        phase = payload["current_phase_rad"]
        if type(amplitude) is not list or type(phase) is not list:
            raise TypeError("AWG current-control snapshot arrays must be JSON arrays")
        return cls(
            system_type=payload["system_type"],
            current_amplitude=tuple(amplitude),
            current_phase_rad=tuple(phase),
        )

    @classmethod
    def from_canonical_json_bytes(cls, payload: bytes) -> "TrackBAWGCurrentControlSnapshotV1":
        decoded = parse_canonical_json_object(
            payload,
            schema="TrackBAWGCurrentControlSnapshotV1",
            exact_fields=frozenset({"schema", "system_type", "current_amplitude", "current_phase_rad"}),
        )
        snapshot = cls.from_dict(decoded)
        require_exact_canonical_json_round_trip(payload, snapshot)
        return snapshot


@dataclass(frozen=True)
class RFCoordinateToAWGRecipeV1:
    """Explicit RF-coordinate to legacy AWG grid/channel adaptation.

    The legacy generators consume one correction value and one phase value per
    grid point.  This recipe is the lossless, ordered adapter: every RF
    coordinate has an explicit grid axis/index, channel set, unit, and current
    control baseline.  It intentionally never reduces a coordinate vector to
    a mean or silently drops an unsupported coordinate kind.
    """

    rf_mapping_core_sha: str
    rf_to_awg_mapping_sha: str
    command_table_sha: str
    command_table_file_sha: str
    channel_mapping_file_sha: str
    generator_source_sha: str
    source_artifact_sha: str
    coordinate_ids: tuple[str, ...]
    coordinate_kinds: tuple[str, ...]
    write_read_roles: tuple[str, ...]
    target_axes: tuple[str, ...]
    target_indices: tuple[int, ...]
    tone_indices: tuple[int, ...]
    target_channels: tuple[tuple[str, ...], ...]
    application_rules: tuple[str, ...]
    units: tuple[str, ...]
    current_amplitude_baseline: tuple[float, ...]
    current_phase_baseline: tuple[float, ...]
    schema: str = field(default="RFCoordinateToAWGRecipeV1", init=False)

    def __post_init__(self) -> None:
        require_sha256(self.rf_mapping_core_sha, "rf_mapping_core_sha")
        for name in (
            "rf_to_awg_mapping_sha", "command_table_sha", "command_table_file_sha",
            "channel_mapping_file_sha", "generator_source_sha", "source_artifact_sha",
        ):
            require_sha256(getattr(self, name), name)
        count = len(self.coordinate_ids)
        if count == 0 or len(set(self.coordinate_ids)) != count:
            raise ValueError("AWG recipe requires unique nonempty RF coordinate IDs")
        fields = (
            self.coordinate_kinds, self.write_read_roles, self.target_axes, self.target_indices,
            self.tone_indices,
            self.target_channels, self.application_rules, self.units,
            self.current_amplitude_baseline, self.current_phase_baseline,
        )
        if any(len(value) != count for value in fields):
            raise ValueError("AWG recipe coordinate fields must have equal lengths")
        for coordinate_id in self.coordinate_ids:
            if type(coordinate_id) is not str or not coordinate_id:
                raise ValueError("AWG recipe coordinate IDs must be nonempty strings")
        for index, kind in enumerate(self.coordinate_kinds):
            if kind not in {"AMPLITUDE", "PHASE"}:
                raise ValueError("AWG recipe coordinate kind is not registered")
            if self.write_read_roles[index] not in {"write", "read", "shared"}:
                raise ValueError("AWG recipe write/read role is not registered")
            if self.target_axes[index] not in AWG_RECIPE_AXES:
                raise ValueError("AWG recipe target axis is not registered")
            if type(self.target_indices[index]) is not int or self.target_indices[index] < 0 \
                    or type(self.tone_indices[index]) is not int or self.tone_indices[index] < 0:
                raise ValueError("AWG recipe target/tone indices must be nonnegative integers")
            channels = tuple(self.target_channels[index])
            if any(channel not in WAVEFORM_CHANNELS for channel in channels):
                raise ValueError("AWG recipe target channel is not registered")
            expected = {
                "AMPLITUDE": ("write", ("W_CH1",), "LOG_AMPLITUDE_TO_W_CH1", "log_amplitude"),
                "PHASE": ("shared", AWG_PHASE_CHANNELS, "PHASE_RAD_TO_Y_CHANNELS", "radian"),
            }[kind]
            if self.write_read_roles[index] != expected[0] or channels != expected[1] \
                    or self.application_rules[index] != expected[2] \
                    or self.units[index] != expected[3]:
                raise ValueError("AWG recipe role/channel/rule/unit does not match coordinate kind")
            if self.tone_indices[index] != self.target_indices[index]:
                raise ValueError("AWG recipe tone index must equal its legacy command-table grid index")
        for name, values in (
            ("current_amplitude_baseline", self.current_amplitude_baseline),
            ("current_phase_baseline", self.current_phase_baseline),
        ):
            if not all(np.isfinite(float(value)) for value in values):
                raise ValueError(f"{name} must contain finite values")

    @classmethod
    def from_core(cls, core: RegisteredRFMappingCoreV1, *, grid_size: int) -> "RFCoordinateToAWGRecipeV1":
        if type(core) is not RegisteredRFMappingCoreV1:
            raise TypeError("AWG recipe requires the exact registered RF mapping core")
        if type(grid_size) is not int or grid_size <= 0:
            raise ValueError("AWG recipe grid size must be a positive integer")
        mapping = core.rf_to_awg_mapping
        if type(mapping) is not RegisteredRFToAWGMappingV1:
            raise ValueError("AWG recipe requires a source-bound RF-to-AWG command-table mapping")
        if mapping.frequency_steps != grid_size:
            raise ValueError("RF-to-AWG mapping grid differs from the selected legacy generator")
        bindings = mapping.bindings
        return cls(
            rf_mapping_core_sha=core.sha256(),
            rf_to_awg_mapping_sha=mapping.sha256(),
            command_table_sha=mapping.command_table_sha,
            command_table_file_sha=mapping.command_table_file_sha,
            channel_mapping_file_sha=mapping.channel_mapping_file_sha,
            generator_source_sha=mapping.generator_source_sha,
            source_artifact_sha=mapping.source_artifact_sha,
            coordinate_ids=tuple(item.coordinate_id for item in bindings),
            coordinate_kinds=tuple(item.kind for item in bindings),
            write_read_roles=tuple(item.write_read_role for item in bindings),
            target_axes=tuple(item.grid_axis for item in bindings),
            target_indices=tuple(item.grid_index for item in bindings),
            tone_indices=tuple(item.tone_index for item in bindings),
            target_channels=tuple(item.awg_channels for item in bindings),
            application_rules=tuple(
                "LOG_AMPLITUDE_TO_W_CH1" if item.application_rule == "ADD_LOG_AMPLITUDE"
                else "PHASE_RAD_TO_Y_CHANNELS"
                for item in bindings
            ),
            units=tuple(item.unit for item in bindings),
            current_amplitude_baseline=tuple(core.current_amplitude),
            current_phase_baseline=tuple(core.current_phase_rad),
        )

    def validate_against(self, core: RegisteredRFMappingCoreV1, *, grid_size: int) -> None:
        if type(core) is not RegisteredRFMappingCoreV1:
            raise TypeError("AWG recipe validation requires the exact registered RF mapping core")
        if self.rf_mapping_core_sha != core.sha256():
            raise ValueError("AWG recipe does not bind the supplied RF mapping core")
        canonical = type(self).from_core(core, grid_size=grid_size)
        if self.to_dict() != canonical.to_dict():
            raise ValueError("AWG recipe is not the exact source-bound command-table mapping")
        if self.current_amplitude_baseline != core.current_amplitude \
                or self.current_phase_baseline != core.current_phase_rad:
            raise ValueError("AWG recipe current-control baselines differ from the RF core")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "rf_mapping_core_sha": self.rf_mapping_core_sha,
            "rf_to_awg_mapping_sha": self.rf_to_awg_mapping_sha,
            "command_table_sha": self.command_table_sha,
            "command_table_file_sha": self.command_table_file_sha,
            "channel_mapping_file_sha": self.channel_mapping_file_sha,
            "generator_source_sha": self.generator_source_sha,
            "source_artifact_sha": self.source_artifact_sha,
            "coordinate_ids": list(self.coordinate_ids),
            "coordinate_kinds": list(self.coordinate_kinds),
            "write_read_roles": list(self.write_read_roles),
            "target_axes": list(self.target_axes),
            "target_indices": list(self.target_indices),
            "tone_indices": list(self.tone_indices),
            "target_channels": [list(value) for value in self.target_channels],
            "application_rules": list(self.application_rules),
            "units": list(self.units),
            "current_amplitude_baseline": list(self.current_amplitude_baseline),
            "current_phase_baseline": list(self.current_phase_baseline),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RFCoordinateToAWGRecipeV1":
        expected = {
            "schema", "rf_mapping_core_sha", "rf_to_awg_mapping_sha", "command_table_sha",
            "command_table_file_sha", "channel_mapping_file_sha", "generator_source_sha",
            "source_artifact_sha", "coordinate_ids", "coordinate_kinds", "write_read_roles",
            "target_axes", "target_indices", "tone_indices", "target_channels",
            "application_rules", "units", "current_amplitude_baseline", "current_phase_baseline",
        }
        if set(payload) != expected or payload.get("schema") != "RFCoordinateToAWGRecipeV1":
            raise ValueError("AWG coordinate recipe schema is invalid")
        return cls(
            rf_mapping_core_sha=str(payload["rf_mapping_core_sha"]),
            rf_to_awg_mapping_sha=str(payload["rf_to_awg_mapping_sha"]),
            command_table_sha=str(payload["command_table_sha"]),
            command_table_file_sha=str(payload["command_table_file_sha"]),
            channel_mapping_file_sha=str(payload["channel_mapping_file_sha"]),
            generator_source_sha=str(payload["generator_source_sha"]),
            source_artifact_sha=str(payload["source_artifact_sha"]),
            coordinate_ids=tuple(str(value) for value in payload["coordinate_ids"]),
            coordinate_kinds=tuple(str(value) for value in payload["coordinate_kinds"]),
            write_read_roles=tuple(str(value) for value in payload["write_read_roles"]),
            target_axes=tuple(str(value) for value in payload["target_axes"]),
            target_indices=tuple(int(value) for value in payload["target_indices"]),
            tone_indices=tuple(int(value) for value in payload["tone_indices"]),
            target_channels=tuple(
                tuple(str(channel) for channel in value) for value in payload["target_channels"]
            ),
            application_rules=tuple(str(value) for value in payload["application_rules"]),
            units=tuple(str(value) for value in payload["units"]),
            current_amplitude_baseline=tuple(float(value) for value in payload["current_amplitude_baseline"]),
            current_phase_baseline=tuple(float(value) for value in payload["current_phase_baseline"]),
        )

    def sha256(self) -> str:
        return registered_rf_to_awg_recipe_sha(
            self.rf_to_awg_mapping_sha,
            rf_mapping_core_sha=self.rf_mapping_core_sha,
            current_amplitude=self.current_amplitude_baseline,
            current_phase_rad=self.current_phase_baseline,
        )

    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_canonical_json_bytes(cls, payload: bytes) -> "RFCoordinateToAWGRecipeV1":
        decoded = parse_canonical_json_object(
            payload,
            schema="RFCoordinateToAWGRecipeV1",
            exact_fields=frozenset({
                "schema", "rf_mapping_core_sha", "rf_to_awg_mapping_sha", "command_table_sha",
                "command_table_file_sha", "channel_mapping_file_sha", "generator_source_sha",
                "source_artifact_sha", "coordinate_ids", "coordinate_kinds", "write_read_roles",
                "target_axes", "target_indices", "tone_indices", "target_channels",
                "application_rules", "units", "current_amplitude_baseline", "current_phase_baseline",
            }),
        )
        recipe = cls.from_dict(decoded)
        require_exact_canonical_json_round_trip(payload, recipe)
        return recipe


def _coordinate_resolved_matrices(
    command: RFCommandDeltaV1,
    core: RegisteredRFMappingCoreV1,
    recipe: RFCoordinateToAWGRecipeV1,
    *,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize ordered RF deltas into the legacy generator's two arrays."""
    recipe.validate_against(core, grid_size=grid_size)
    expected_ids = tuple(item.coordinate_id for item in core.rf_coordinates)
    if type(command) is not RFCommandDeltaV1 or command.ordered_tone_ids != expected_ids:
        raise ValueError("RF command coordinates must exactly match the RF core ordering")
    if command.rf_coordinate_layout_sha != core.rf_coordinate_layout_sha:
        raise ValueError("RF command layout differs from the RF core")
    if not core.phase_wrap_valid(command.phase_delta_rad):
        raise ValueError("RF command phase delta violates the core wrap convention")
    log_amplitude = np.zeros((grid_size, grid_size), dtype=np.float64)
    phase = np.zeros((grid_size, grid_size), dtype=np.float64)
    for index, kind in enumerate(recipe.coordinate_kinds):
        axis = recipe.target_axes[index]
        target_index = recipe.target_indices[index]
        selector = (slice(None), slice(None))
        if axis == "row":
            selector = (target_index, slice(None))
        else:
            selector = (slice(None), target_index)
        if kind == "AMPLITUDE":
            log_amplitude[selector] += (
                recipe.current_amplitude_baseline[index] + command.amplitude_delta[index]
            )
        elif kind == "PHASE":
            phase[selector] += (
                recipe.current_phase_baseline[index] + command.phase_delta_rad[index]
            )
        elif abs(float(command.other_delta[index])) > 1.0e-15:
            raise ValueError("RF command contains an unsupported nonzero OTHER coordinate")
    return np.exp(log_amplitude), phase


def _spiral_order(size: int) -> tuple[int, ...]:
    """Match the nested spiral order used by awg225seq/awg256seq."""
    center = (size - 1) // 2
    coordinates = [(center, center)]
    step = 1
    directions = ((0, 1), (1, 0), (0, -1), (-1, 0))
    row, column, direction = center, center, 0
    while len(coordinates) < size * size:
        for _ in range(2):
            delta_row, delta_column = directions[direction % 4]
            for _ in range(step):
                row += delta_row
                column += delta_column
                if 0 <= row < size and 0 <= column < size:
                    coordinates.append((row, column))
                    if len(coordinates) >= size * size:
                        break
            direction += 1
            if len(coordinates) >= size * size:
                break
        step += 1
    return tuple(row * size + column for row, column in coordinates)


def _scan_orders(size: int) -> dict[str, tuple[int, ...]]:
    odd = tuple(range(0, size * size, 2))
    even = tuple(range(1, size * size, 2))
    return {"regular": odd + even, "linear": tuple(range(size * size)), "spiral": _spiral_order(size)}


def _runtime_identity() -> str:
    matplotlib_version = "unavailable"
    try:
        matplotlib_version = str(metadata.version("matplotlib"))
    except metadata.PackageNotFoundError:
        pass
    return canonical_sha256({
        "schema": "TrackBAWGReproductionRuntimeV1",
        "python": sys.version,
        "numpy": np.__version__,
        "matplotlib": matplotlib_version,
    })


def _action_payload(
    system_type: str, amplitude_coordinates: Sequence[float], phase_coordinates: Sequence[float],
) -> dict[str, object]:
    return {
        "schema": "TrackBLogicalActionPayloadV1",
        "system_type": system_type,
        "amplitude_coordinates": [float(value) for value in amplitude_coordinates],
        "phase_coordinates": [float(value) for value in phase_coordinates],
        "coordinate_layout_sha": coordinate_layout_sha(3 if system_type == "3qubit" else 4),
    }


def _assert_proposal_receipt_lineage(
    proposal: ResearchRFActionProposalV1,
    eligibility_receipt: TrackBProposalEligibilityReceiptV1,
) -> None:
    """Require the AWG proposal to retain its receipt's upstream lineage."""
    for field_name in (
        "posterior_receipt_sha",
        "applied_evaluation_sha",
        "eligibility_policy_sha",
    ):
        if getattr(proposal, field_name) != getattr(eligibility_receipt, field_name):
            raise ValueError(
                f"AWG reproduction proposal {field_name} differs from its eligibility receipt"
            )


@dataclass(frozen=True)
class TrackBAWGReproductionInputV1:
    """Source-bound action and explicit write policy for one AWG reproduction."""

    system_type: str
    amplitude_coordinates: tuple[float, ...]
    phase_coordinates: tuple[float, ...]
    logical_action_sha: str
    source_commit: str
    source_tree: str
    output_root: str | None = None
    write_files: bool = False
    no_upload: bool = True
    proposal: ResearchRFActionProposalV1 | None = None
    eligibility_receipt: TrackBProposalEligibilityReceiptV1 | None = None
    rf_command_delta: RFCommandDeltaV1 | None = None
    rf_mapping_core: RegisteredRFMappingCoreV1 | None = None
    coordinate_recipe: RFCoordinateToAWGRecipeV1 | None = None
    external_artifact_file_shas: tuple[tuple[str, str], ...] = ()
    external_artifact_paths: tuple[tuple[str, str], ...] = ()
    schema: str = field(default="TrackBAWGReproductionInputV1", init=False)

    def __post_init__(self) -> None:
        if self.system_type not in SYSTEM_GENERATOR_MODULES:
            raise ValueError("Track B AWG reproduction system type is unsupported")
        expected = 2 if self.system_type == "3qubit" else 3
        for values, name in (
            (self.amplitude_coordinates, "amplitude_coordinates"),
            (self.phase_coordinates, "phase_coordinates"),
        ):
            if len(values) != expected or not all(np.isfinite(float(value)) for value in values):
                raise ValueError(f"{name} must contain {expected} finite values")
        payload = _action_payload(self.system_type, self.amplitude_coordinates, self.phase_coordinates)
        if self.logical_action_sha != canonical_sha256(payload):
            raise ValueError("logical_action_sha does not bind the canonical Track B action payload")
        require_sha256(self.logical_action_sha, "logical_action_sha")
        require_git_oid(self.source_commit, "source_commit")
        require_git_oid(self.source_tree, "source_tree")
        if self.no_upload is not True:
            raise ValueError("Track B AWG reproduction cannot enable upload")
        if self.external_artifact_file_shas or self.external_artifact_paths:
            if tuple(name for name, _sha in self.external_artifact_file_shas) != BOUND_AWG_ARTIFACT_NAMES \
                    or tuple(name for name, _path in self.external_artifact_paths) != BOUND_AWG_ARTIFACT_NAMES:
                raise ValueError("external AWG artifact file bindings must be complete and ordered")
            for _name, value in self.external_artifact_file_shas:
                require_sha256(value, "external_artifact_file_sha")
            for _name, value in self.external_artifact_paths:
                if type(value) is not str:
                    raise ValueError("external AWG artifact paths must be existing absolute files")
                artifact_path = Path(value)
                if not artifact_path.is_absolute() or not artifact_path.is_file():
                    raise ValueError("external AWG artifact paths must be existing absolute files")
        if self.write_files and self.proposal is not None and not self.external_artifact_file_shas:
            raise ValueError("written bound AWG reproduction requires rehashable source artifact files")
        chain = (
            self.proposal, self.eligibility_receipt,
            self.rf_command_delta, self.rf_mapping_core,
        )
        if any(value is not None for value in chain) and not all(value is not None for value in chain):
            raise ValueError("complete Track B AWG reproduction binding requires proposal, receipt, RF delta, and core")
        if self.proposal is not None:
            if type(self.proposal) is not ResearchRFActionProposalV1 \
                    or type(self.eligibility_receipt) is not TrackBProposalEligibilityReceiptV1 \
                    or type(self.rf_command_delta) is not RFCommandDeltaV1 \
                    or type(self.rf_mapping_core) is not RegisteredRFMappingCoreV1:
                raise TypeError("Track B AWG reproduction requires exact proposal, receipt, RF delta, and core types")
            if self.coordinate_recipe is None:
                object.__setattr__(
                    self, "coordinate_recipe",
                    RFCoordinateToAWGRecipeV1.from_core(
                        self.rf_mapping_core, grid_size=SYSTEM_GRID_STEPS[self.system_type],
                    ),
                )
            if type(self.coordinate_recipe) is not RFCoordinateToAWGRecipeV1:
                raise TypeError("Track B AWG reproduction requires the exact coordinate recipe type")
            self.coordinate_recipe.validate_against(
                self.rf_mapping_core, grid_size=SYSTEM_GRID_STEPS[self.system_type],
            )
            self.proposal.validate()
            self.eligibility_receipt.validate()
            if self.proposal.source_commit != self.source_commit or self.proposal.source_tree != self.source_tree:
                raise ValueError("AWG reproduction source identity differs from proposal")
            if self.proposal.eligibility_receipt_sha != self.eligibility_receipt.sha256():
                raise ValueError("AWG reproduction proposal does not bind the eligibility receipt")
            _assert_proposal_receipt_lineage(self.proposal, self.eligibility_receipt)
            if self.eligibility_receipt.applied_candidate_sha != self.proposal.selected_candidate_sha:
                raise ValueError("AWG reproduction proposal candidate differs from eligibility receipt")
            if self.eligibility_receipt.rf_command_delta_sha != self.rf_command_delta.sha256():
                raise ValueError("AWG reproduction RF command differs from eligibility receipt")
            if self.eligibility_receipt.rf_mapping_core_sha != self.rf_mapping_core.sha256():
                raise ValueError("AWG reproduction RF core differs from eligibility receipt")
            if self.rf_command_delta.rf_coordinate_layout_sha != self.rf_mapping_core.rf_coordinate_layout_sha:
                raise ValueError("AWG reproduction RF coordinate layout differs from RF core")
            if self.rf_command_delta.system_type != self.system_type \
                    or self.rf_command_delta.ordered_tone_ids != self.coordinate_recipe.coordinate_ids:
                raise ValueError("AWG reproduction RF command ordering differs from the coordinate recipe")
            if self.rf_command_delta.logical_action_payload_sha != self.proposal.action_sha:
                raise ValueError("AWG reproduction RF delta does not bind the proposal logical action")
            if self.proposal.current_control_snapshot_sha != self.rf_mapping_core.current_control_snapshot_sha:
                raise ValueError("AWG reproduction current-control snapshot differs from RF core")
            if self.proposal.bounds_policy_sha != self.rf_mapping_core.bounds_policy_sha \
                    or self.proposal.rf_coordinate_layout_sha != self.rf_mapping_core.rf_coordinate_layout_sha \
                    or self.proposal.rf_command_basis_sha != self.rf_mapping_core.rf_command_basis_sha \
                    or self.proposal.logical_coordinate_layout_sha != self.rf_mapping_core.logical_coordinate_layout_sha:
                raise ValueError("AWG reproduction proposal ABI identities differ from RF core")
            if any(value is None for value in (
                self.rf_mapping_core.operation_registry_sha,
                self.rf_mapping_core.command_table_sha,
                self.rf_mapping_core.hardware_path_id,
                self.rf_mapping_core.operating_point_id,
            )):
                raise ValueError("complete AWG reproduction requires RF-core operation/hardware identities")
            if self.proposal.operation_registry_sha != self.rf_mapping_core.operation_registry_sha \
                    or self.proposal.command_table_sha != self.rf_mapping_core.command_table_sha \
                    or self.proposal.hardware_path_id != self.rf_mapping_core.hardware_path_id \
                    or self.proposal.operating_point_id != self.rf_mapping_core.operating_point_id:
                raise ValueError("AWG reproduction proposal operation/hardware identities differ from RF core")
            if self.rf_mapping_core.rf_to_awg_mapping is None \
                    or self.rf_mapping_core.rf_to_awg_mapping.sha256() != self.proposal.rf_to_awg_mapping_sha \
                    or self.coordinate_recipe.sha256() != self.proposal.awg_recipe_sha \
                    or self.coordinate_recipe.sha256() != self.eligibility_receipt.awg_recipe_sha \
                    or self.proposal.action_sha != self.eligibility_receipt.logical_action_payload_sha:
                raise ValueError("AWG reproduction proposal/receipt do not bind the source-bound recipe/action")
            if any(abs(float(value)) > 1.0e-15 for value in (self.rf_command_delta.other_delta or ())):
                raise ValueError("legacy AWG reproduction cannot silently drop non-amplitude/phase RF commands")
        if self.write_files:
            if type(self.output_root) is not str or not Path(self.output_root).is_absolute():
                raise ValueError("explicit AWG reproduction writes require an absolute output root")
        elif self.output_root is not None:
            if type(self.output_root) is not str or not Path(self.output_root).is_absolute():
                raise ValueError("preview output_root, when supplied, must be absolute")

    @classmethod
    def from_grouped_action(
        cls,
        *,
        system_type: str,
        amplitude_coordinates: Sequence[float],
        phase_coordinates: Sequence[float],
        source_commit: str,
        source_tree: str,
        output_root: str | None = None,
        write_files: bool = False,
    ) -> "TrackBAWGReproductionInputV1":
        payload = _action_payload(system_type, amplitude_coordinates, phase_coordinates)
        return cls(
            system_type=system_type,
            amplitude_coordinates=tuple(float(value) for value in amplitude_coordinates),
            phase_coordinates=tuple(float(value) for value in phase_coordinates),
            logical_action_sha=canonical_sha256(payload),
            source_commit=source_commit,
            source_tree=source_tree,
            output_root=output_root,
            write_files=write_files,
        )

    @classmethod
    def from_bound_proposal(
        cls,
        *, proposal: ResearchRFActionProposalV1,
        eligibility_receipt: TrackBProposalEligibilityReceiptV1,
        rf_command_delta: RFCommandDeltaV1,
        rf_mapping_core: RegisteredRFMappingCoreV1,
        output_root: str,
        write_files: bool = True,
        external_artifact_file_shas: tuple[tuple[str, str], ...] = (),
        external_artifact_paths: tuple[tuple[str, str], ...] = (),
    ) -> "TrackBAWGReproductionInputV1":
        proposal.validate()
        logical = proposal.logical_action
        return cls(
            system_type=proposal.system_type,
            amplitude_coordinates=tuple(float(value) for value in logical["amplitude_coordinates"]),
            phase_coordinates=tuple(float(value) for value in logical["phase_coordinates"]),
            logical_action_sha=proposal.action_sha,
            source_commit=proposal.source_commit,
            source_tree=proposal.source_tree,
            output_root=output_root,
            write_files=write_files,
            proposal=proposal,
            eligibility_receipt=eligibility_receipt,
            rf_command_delta=rf_command_delta,
            rf_mapping_core=rf_mapping_core,
            external_artifact_file_shas=external_artifact_file_shas,
            external_artifact_paths=external_artifact_paths,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "system_type": self.system_type,
            "amplitude_coordinates": list(self.amplitude_coordinates),
            "phase_coordinates": list(self.phase_coordinates),
            "logical_action_sha": self.logical_action_sha,
            "source_commit": self.source_commit,
            "source_tree": self.source_tree,
            "write_files": self.write_files,
            "no_upload": True,
            "proposal_sha": None if self.proposal is None else self.proposal.sha256(),
            "eligibility_receipt_sha": (
                None if self.eligibility_receipt is None else self.eligibility_receipt.sha256()
            ),
            "rf_command_delta_sha": (
                None if self.rf_command_delta is None else self.rf_command_delta.sha256()
            ),
            "rf_mapping_core_sha": (
                None if self.rf_mapping_core is None else self.rf_mapping_core.sha256()
            ),
            "coordinate_recipe_sha": (
                None if self.coordinate_recipe is None else self.coordinate_recipe.sha256()
            ),
            "coordinate_recipe": (
                None if self.coordinate_recipe is None else self.coordinate_recipe.to_dict()
            ),
            "external_artifact_file_shas": [list(item) for item in self.external_artifact_file_shas],
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class TrackBAWGReproductionReceiptV1:
    """Hash receipt for one preview or explicit isolated generator run."""

    input_sha: str
    generator_module: str
    generator_source_sha: str
    generator_config_sha: str
    runtime_sha: str
    command_table_sha: str
    matrix_sha: str
    waveform_bundle_sha: str | None
    diagnostic_bundle_sha: str
    fourier_report_sha: str
    output_root: str | None
    relative_waveform_paths: tuple[str, ...]
    waveform_file_manifest_sha: str | None
    write_files: bool
    relative_output_paths: tuple[str, ...] = ()
    output_file_manifest_sha: str | None = None
    proposal_sha: str | None = None
    eligibility_receipt_sha: str | None = None
    rf_command_delta_sha: str | None = None
    rf_mapping_core_sha: str | None = None
    coordinate_recipe_sha: str | None = None
    transport_dependency_audit_sha: str | None = None
    transport_dependency_status: str = "NO_UPLOAD_DEPENDENCY_IN_INSPECTED_SOURCE"
    schema: str = field(default="TrackBAWGReproductionReceiptV1", init=False)

    def __post_init__(self) -> None:
        for name in (
            "input_sha", "generator_source_sha", "generator_config_sha", "runtime_sha",
            "command_table_sha", "matrix_sha", "diagnostic_bundle_sha", "fourier_report_sha",
        ):
            require_sha256(getattr(self, name), name)
        if self.waveform_bundle_sha is not None:
            require_sha256(self.waveform_bundle_sha, "waveform_bundle_sha")
        if self.transport_dependency_audit_sha is not None:
            require_sha256(self.transport_dependency_audit_sha, "transport_dependency_audit_sha")
        if self.write_files:
            if self.waveform_bundle_sha is None or not self.relative_waveform_paths \
                    or self.waveform_file_manifest_sha is None \
                    or not self.relative_output_paths or self.output_file_manifest_sha is None:
                raise ValueError("written AWG reproduction requires waveform files and a bundle hash")
            if type(self.output_root) is not str or not Path(self.output_root).is_absolute():
                raise ValueError("written AWG reproduction requires an absolute output root")
            for name in (
                "proposal_sha", "eligibility_receipt_sha", "rf_command_delta_sha",
                "rf_mapping_core_sha", "coordinate_recipe_sha",
                "transport_dependency_audit_sha",
                "output_file_manifest_sha",
            ):
                require_sha256(getattr(self, name), name)
        elif self.waveform_bundle_sha is not None or self.relative_waveform_paths \
                or self.waveform_file_manifest_sha is not None or self.relative_output_paths \
                or self.output_file_manifest_sha is not None or self.output_root is not None:
            raise ValueError("preview-only AWG reproduction cannot claim written waveform files")
        if self.transport_dependency_status != "NO_UPLOAD_DEPENDENCY_IN_INSPECTED_SOURCE":
            raise ValueError("AWG reproduction dependency observation is not registered")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "input_sha": self.input_sha,
            "generator_module": self.generator_module,
            "generator_source_sha": self.generator_source_sha,
            "generator_config_sha": self.generator_config_sha,
            "runtime_sha": self.runtime_sha,
            "command_table_sha": self.command_table_sha,
            "matrix_sha": self.matrix_sha,
            "waveform_bundle_sha": self.waveform_bundle_sha,
            "diagnostic_bundle_sha": self.diagnostic_bundle_sha,
            "fourier_report_sha": self.fourier_report_sha,
            "output_root": self.output_root,
            "relative_waveform_paths": list(self.relative_waveform_paths),
            "waveform_file_manifest_sha": self.waveform_file_manifest_sha,
            "relative_output_paths": list(self.relative_output_paths),
            "output_file_manifest_sha": self.output_file_manifest_sha,
            "write_files": self.write_files,
            "proposal_sha": self.proposal_sha,
            "eligibility_receipt_sha": self.eligibility_receipt_sha,
            "rf_command_delta_sha": self.rf_command_delta_sha,
            "rf_mapping_core_sha": self.rf_mapping_core_sha,
            "coordinate_recipe_sha": self.coordinate_recipe_sha,
            "transport_dependency_audit_sha": self.transport_dependency_audit_sha,
            "transport_dependency_status": self.transport_dependency_status,
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class TrackBAWGReproductionPairReceiptV1:
    """Evidence that two isolated explicit runs produced identical bytes."""

    first_receipt_sha: str
    second_receipt_sha: str
    command_table_sha: str
    matrix_sha: str
    waveform_bundle_sha: str
    waveform_file_manifest_sha: str
    diagnostic_bundle_sha: str
    fourier_report_sha: str
    exact_match: bool = True
    schema: str = field(default="TrackBAWGReproductionPairReceiptV1", init=False)

    def __post_init__(self) -> None:
        for name in (
            "first_receipt_sha", "second_receipt_sha", "command_table_sha", "matrix_sha",
            "waveform_bundle_sha", "diagnostic_bundle_sha", "fourier_report_sha",
            "waveform_file_manifest_sha",
        ):
            require_sha256(getattr(self, name), name)
        if self.exact_match is not True:
            raise ValueError("a reproduction pair receipt is valid only for an exact match")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "first_receipt_sha": self.first_receipt_sha,
            "second_receipt_sha": self.second_receipt_sha,
            "command_table_sha": self.command_table_sha,
            "matrix_sha": self.matrix_sha,
            "waveform_bundle_sha": self.waveform_bundle_sha,
            "waveform_file_manifest_sha": self.waveform_file_manifest_sha,
            "diagnostic_bundle_sha": self.diagnostic_bundle_sha,
            "fourier_report_sha": self.fourier_report_sha,
            "exact_match": True,
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def _build_recipe(
    reproduction_input: TrackBAWGReproductionInputV1,
) -> tuple[
    LegacyAWGGeneratorManifestV1,
    dict[str, object],
    np.ndarray,
    np.ndarray,
    str,
    str,
    str,
    str,
    str,
    str,
    bytes,
]:
    """Build a pure matrix/command recipe after auditing the static source manifest."""
    generator_manifest = LEGACY_AWG_GENERATOR_MANIFESTS[reproduction_input.system_type]
    generator_source_bytes = generator_manifest.source_path.read_bytes()
    transport_observation, transport_audit_sha = _assert_no_transport_dependency(
        generator_manifest,
        generator_source_bytes,
    )
    size = generator_manifest.frequency_steps
    rows = np.arange(size, dtype=np.int64)[:, None]
    columns = np.arange(size, dtype=np.int64)[None, :]
    if reproduction_input.rf_command_delta is not None:
        command = reproduction_input.rf_command_delta
        recipe = reproduction_input.coordinate_recipe
        if recipe is None or reproduction_input.rf_mapping_core is None:
            raise ValueError("bound RF reproduction requires its coordinate-resolved AWG recipe")
        correction_matrix, phase_matrix = _coordinate_resolved_matrices(
            command, reproduction_input.rf_mapping_core, recipe, grid_size=size,
        )
        matrix_recipe = recipe.to_dict()
    else:
        mode_coordinates = grouped_to_mode_action(
            reproduction_input.amplitude_coordinates,
            reproduction_input.phase_coordinates,
            3 if reproduction_input.system_type == "3qubit" else 4,
        )
        mode_amplitude = np.asarray([pair[0] for pair in mode_coordinates], dtype=np.float64)
        mode_phase = np.asarray([pair[1] for pair in mode_coordinates], dtype=np.float64)
        correction_matrix = np.exp(
            mode_amplitude[rows % len(mode_amplitude)]
            + mode_amplitude[columns % len(mode_amplitude)]
        )
        phase_matrix = (
            mode_phase[rows % len(mode_phase)]
            + mode_phase[columns % len(mode_phase)]
        )
        matrix_recipe = "exp(mode_amplitude[row] + mode_amplitude[column]) and mode_phase[row] + mode_phase[column]"
    correction_matrix = np.asarray(correction_matrix, dtype=np.float64)
    phase_matrix = np.asarray(phase_matrix, dtype=np.float64)
    matrix_sha = canonical_sha256({
        "schema": "TrackBAWGActionMatrixRecipeV1",
        "correction_matrix_sha": array_identity_sha256(correction_matrix),
        "phase_matrix_sha": array_identity_sha256(phase_matrix),
        "matrix_recipe": matrix_recipe,
        "coordinate_recipe_sha": (
            None if reproduction_input.coordinate_recipe is None
            else reproduction_input.coordinate_recipe.sha256()
        ),
        "coordinate_recipe": (
            None if reproduction_input.coordinate_recipe is None
            else reproduction_input.coordinate_recipe.to_dict()
        ),
        "rf_command_delta_sha": (
            None if reproduction_input.rf_command_delta is None
            else reproduction_input.rf_command_delta.sha256()
        ),
    })
    frequencies = np.linspace(
        generator_manifest.frequency_start_mhz,
        generator_manifest.frequency_end_mhz,
        size,
    )
    orders = _scan_orders(size)
    command_rows = []
    for point_index in range(size * size):
        row_index, column_index = divmod(point_index, size)
        command_rows.append({
            "point_index": point_index,
            "grid_row": row_index,
            "grid_column": column_index,
            "x_frequency_mhz": float(frequencies[row_index]),
            "y_frequency_mhz": float(frequencies[column_index]),
            "regular_position": int(orders["regular"].index(point_index)),
            "linear_position": int(orders["linear"].index(point_index)),
            "spiral_position": int(orders["spiral"].index(point_index)),
        })
    command_table = {
        "schema": "TrackBAWGCommandTableV1",
        "system_type": reproduction_input.system_type,
        "frequency_start_mhz": generator_manifest.frequency_start_mhz,
        "frequency_end_mhz": generator_manifest.frequency_end_mhz,
        "frequency_steps": size,
        "sampling_rate_hz": generator_manifest.sampling_rate_hz,
        "point_duration_ns": generator_manifest.point_duration_ns,
        "channel_analog_delays": {
            str(key): int(value) for key, value in generator_manifest.channel_analog_delays.items()
        },
        "channels": list(WAVEFORM_CHANNELS),
        "proposal_sha": (
            None if reproduction_input.proposal is None else reproduction_input.proposal.sha256()
        ),
        "proposal_command_table_sha": (
            None if reproduction_input.proposal is None else reproduction_input.proposal.command_table_sha
        ),
        "proposal_operation_registry_sha": (
            None if reproduction_input.proposal is None else reproduction_input.proposal.operation_registry_sha
        ),
        "hardware_path_id": (
            None if reproduction_input.proposal is None else reproduction_input.proposal.hardware_path_id
        ),
        "operating_point_id": (
            None if reproduction_input.proposal is None else reproduction_input.proposal.operating_point_id
        ),
        "eligibility_receipt_sha": (
            None if reproduction_input.eligibility_receipt is None
            else reproduction_input.eligibility_receipt.sha256()
        ),
        "rf_command_delta_sha": (
            None if reproduction_input.rf_command_delta is None
            else reproduction_input.rf_command_delta.sha256()
        ),
        "rf_mapping_core_sha": (
            None if reproduction_input.rf_mapping_core is None
            else reproduction_input.rf_mapping_core.sha256()
        ),
        "coordinate_recipe_sha": (
            None if reproduction_input.coordinate_recipe is None
            else reproduction_input.coordinate_recipe.sha256()
        ),
        "scan_orders": {name: list(order) for name, order in orders.items()},
        "rows": command_rows,
    }
    command_table_sha = canonical_sha256(command_table)
    config_payload = generator_manifest.static_config_payload()
    config_payload.update({
        "matrix_sha": matrix_sha,
        "command_table_sha": command_table_sha,
        "generator_module": generator_manifest.module_name,
        "recipe_version": "TrackBAWGActionMatrixRecipeV1",
    })
    generator_config_sha = canonical_sha256(config_payload)
    generator_source_sha = generator_manifest.source_sha256
    if reproduction_input.rf_mapping_core is not None:
        source_mapping = reproduction_input.rf_mapping_core.rf_to_awg_mapping
        if source_mapping is None:
            raise ValueError("bound AWG reproduction requires a source-bound RF-to-AWG mapping")
        if source_mapping.generator_source_sha != generator_source_sha \
                or source_mapping.frequency_steps != size:
            raise ValueError("RF-to-AWG mapping does not match the loaded legacy generator source/configuration")
    return (
        generator_manifest, command_table, correction_matrix, phase_matrix,
        command_table_sha, matrix_sha, generator_config_sha, generator_source_sha,
        transport_observation, transport_audit_sha, generator_source_bytes,
    )


def _preview_diagnostic(
    reproduction_input: TrackBAWGReproductionInputV1,
    command_table: Mapping[str, object],
    correction_matrix: np.ndarray,
    phase_matrix: np.ndarray,
    *,
    status: str,
) -> dict[str, object]:
    return {
        "schema": "TrackBAWGDiagnosticReportV1",
        "status": status,
        "system_type": reproduction_input.system_type,
        "action_sha": reproduction_input.logical_action_sha,
        "command_table_sha": canonical_sha256(command_table),
        "correction_matrix_sha": array_identity_sha256(correction_matrix),
        "phase_matrix_sha": array_identity_sha256(phase_matrix),
        "fourier_guard": {
            "status": "PENDING_WRITTEN_WAVEFORM_ANALYSIS" if status != "WRITTEN" else "SEE_GENERATED_REPORT",
            "sampling_rate_hz": float(command_table["sampling_rate_hz"]),
            "frequency_grid_mhz": [
                float(value) for value in np.linspace(
                    float(command_table["frequency_start_mhz"]),
                    float(command_table["frequency_end_mhz"]),
                    int(command_table["frequency_steps"]),
                )
            ],
        },
        "power_and_leakage": {
            "status": "PENDING_WRITTEN_WAVEFORM_ANALYSIS" if status != "WRITTEN" else "SEE_GENERATED_REPORT",
        },
    }


_AUDITED_LEGACY_GENERATOR_HARNESS = r'''
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys

import numpy as np

module_name, expected_source_path, expected_source_sha, output_root, correction_path, phase_path, manifest_json = sys.argv[1:]
expected_manifest = json.loads(manifest_json)
os.environ["MPLBACKEND"] = "Agg"
os.environ["AWG_SHOW_PLOTS"] = "0"
if expected_manifest["module_name"] != module_name:
    raise RuntimeError("legacy generator module identity differs from its static manifest")
source_path = Path(expected_source_path).resolve()
if not source_path.is_file():
    raise RuntimeError("legacy generator source path is missing")
source_text = source_path.read_text(encoding="utf-8")
source_sha = hashlib.sha256(source_text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")).hexdigest()
if source_sha != expected_source_sha:
    raise RuntimeError("legacy generator source hash changed after static audit")
spec = importlib.util.spec_from_file_location(f"trackb_audited_{expected_source_sha[:16]}", source_path)
if spec is None or spec.loader is None:
    raise RuntimeError("legacy generator source cannot be loaded from its audited path")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
if Path(module.__file__).resolve() != source_path:
    raise RuntimeError("legacy generator loaded an unexpected source file")
config = module.AODAWGConfig()
actual = {
    "frequency_start_mhz": float(config.freq_start),
    "frequency_end_mhz": float(config.freq_end),
    "frequency_steps": int(config.freq_steps),
    "sampling_rate_hz": float(config.sampling_rate),
    "point_duration_ns": int(config.point_duration),
    "channel_analog_delays": {str(key): int(value) for key, value in config.channel_analog_delays.items()},
}
expected = {
    "frequency_start_mhz": expected_manifest["frequency_start_mhz"],
    "frequency_end_mhz": expected_manifest["frequency_end_mhz"],
    "frequency_steps": expected_manifest["frequency_steps"],
    "sampling_rate_hz": expected_manifest["sampling_rate_hz"],
    "point_duration_ns": expected_manifest["point_duration_ns"],
    "channel_analog_delays": expected_manifest["channel_analog_delays"],
}
if actual != expected:
    raise RuntimeError("legacy generator configuration differs from its registered static manifest")
config.output_dir = output_root
config.correction_matrix = np.load(correction_path, allow_pickle=False).reshape(-1)
config.phase_vector = np.load(phase_path, allow_pickle=False).reshape(-1)
with contextlib.redirect_stdout(io.StringIO()):
    module.generate_awg_sequence(config)
'''


def _execute_audited_legacy_generator(
    generator_manifest: LegacyAWGGeneratorManifestV1,
    *,
    generator_source_bytes: bytes,
    output_root: Path,
    correction_matrix: np.ndarray,
    phase_matrix: np.ndarray,
) -> None:
    """Execute only a pre-audited generator in a fresh subprocess boundary."""
    with tempfile.TemporaryDirectory(prefix="trackb_awg_generator_") as temporary_directory:
        temporary_root = Path(temporary_directory)
        source_path = temporary_root / "audited_legacy_generator.py"
        correction_path = temporary_root / "correction_matrix.npy"
        phase_path = temporary_root / "phase_matrix.npy"
        with source_path.open("xb") as handle:
            handle.write(generator_source_bytes)
        np.save(correction_path, np.asarray(correction_matrix, dtype=np.float64), allow_pickle=False)
        np.save(phase_path, np.asarray(phase_matrix, dtype=np.float64), allow_pickle=False)
        command = [
            sys.executable,
            "-I",
            "-c",
            _AUDITED_LEGACY_GENERATOR_HARNESS,
            generator_manifest.module_name,
            str(source_path),
            generator_manifest.source_sha256,
            str(output_root),
            str(correction_path),
            str(phase_path),
            canonical_json_bytes(generator_manifest.static_config_payload()).decode("utf-8"),
        ]
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[3],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    if completed.returncode != 0:
        diagnostic = (completed.stderr or completed.stdout or "legacy generator subprocess failed").strip()
        raise RuntimeError(f"audited legacy AWG generator failed: {diagnostic[:2000]}")


def _project_amplitude(samples: np.ndarray, frequency_mhz: float, sampling_rate: float) -> float:
    signal = np.asarray(samples, dtype=np.float64)
    signal = np.nan_to_num(signal, copy=False)
    signal = signal - float(np.mean(signal))
    time = np.arange(signal.size, dtype=np.float64) / sampling_rate
    return float(2.0 * abs(np.dot(signal, np.exp(-2j * np.pi * frequency_mhz * 1.0e6 * time))) / max(signal.size, 1))


def _analyze_generated_waveforms(
    reproduction_input: TrackBAWGReproductionInputV1,
    command_table: Mapping[str, object],
    output_root: Path,
) -> dict[str, object]:
    size = int(command_table["frequency_steps"])
    frequencies = np.linspace(
        float(command_table["frequency_start_mhz"]),
        float(command_table["frequency_end_mhz"]), size,
    )
    orders = _scan_orders(size)
    variant_arrays: dict[str, dict[str, np.ndarray]] = {}
    for variant in WAVEFORM_VARIANTS:
        variant_arrays[variant] = {}
        for channel in WAVEFORM_CHANNELS:
            channel_files = tuple(sorted(output_root.glob(f"AWG_{channel}_*.txt")))
            matches = (
                tuple(
                    path for path in channel_files
                    if "_linear_" not in path.name and "_spiral_" not in path.name
                )
                if variant == "regular" else tuple(
                    path for path in channel_files if f"_{variant}_" in path.name
                )
            )
            if len(matches) != 1:
                raise ValueError(f"written AWG reproduction has an ambiguous {variant} {channel} file")
            variant_arrays[variant][channel] = np.loadtxt(matches[0], delimiter=",", skiprows=1)
    window_reports = []
    power_summary: dict[str, dict[str, float]] = {}
    for variant, arrays in variant_arrays.items():
        positions = orders[variant]
        for channel in WAVEFORM_CHANNELS:
            data = np.asarray(arrays[channel])
            delay = int(command_table["channel_analog_delays"][channel])
            # The legacy generator stores channel analog delay in the output
            # index; use the registered config value rather than guessing it
            # from the first nonzero sample.
            powers = []
            leakage_ratios = []
            expected_amplitudes = []
            for position, point_index in enumerate(positions):
                row_index, column_index = divmod(int(point_index), size)
                start = position * int(command_table["point_duration_ns"]) + delay
                duration = int(command_table["point_duration_ns"])
                segment = data[start:start + duration, 0]
                if segment.size != duration:
                    raise ValueError("generated AWG waveform is shorter than its registered command table")
                frequency_index = row_index if channel.endswith(("CH1", "CH3")) else column_index
                expected = float(frequencies[frequency_index])
                expected_amplitude = _project_amplitude(
                    segment, expected, float(command_table["sampling_rate_hz"]),
                )
                spectrum = tuple(
                    _project_amplitude(segment, float(frequency), float(command_table["sampling_rate_hz"]))
                    for frequency in frequencies
                )
                outside = max(
                    (value for index, value in enumerate(spectrum) if index != frequency_index),
                    default=0.0,
                )
                leakage_ratio = float(outside / max(expected_amplitude, 1.0e-12))
                powers.append(float(np.mean(np.square(segment))))
                leakage_ratios.append(leakage_ratio)
                expected_amplitudes.append(expected_amplitude)
            power_summary[f"{variant}:{channel}"] = {
                "mean_power": float(np.mean(powers)),
                "max_power": float(np.max(powers)),
                "min_expected_amplitude": float(np.min(expected_amplitudes)),
                "max_grid_leakage_ratio": float(np.max(leakage_ratios)),
            }
            window_reports.append({
                "variant": variant,
                "channel": channel,
                "windows": len(positions),
                "max_grid_leakage_ratio": float(np.max(leakage_ratios)),
                "min_expected_amplitude": float(np.min(expected_amplitudes)),
            })
    return {
        "schema": "TrackBAWGDiagnosticReportV1",
        "status": "WRITTEN",
        "system_type": reproduction_input.system_type,
        "action_sha": reproduction_input.logical_action_sha,
        "fourier_guard": {
            "status": "ANALYZED_ON_REGISTERED_FREQUENCY_GRID",
            "sampling_rate_hz": float(command_table["sampling_rate_hz"]),
            "frequency_grid_mhz": [float(value) for value in frequencies],
            "windows": window_reports,
        },
        "power_and_leakage": power_summary,
    }


def preview_trackb_awg_reproduction(
    reproduction_input: TrackBAWGReproductionInputV1,
) -> TrackBAWGReproductionReceiptV1:
    """Validate and hash the reproduction recipe without creating any files."""
    if reproduction_input.write_files:
        raise ValueError("preview_trackb_awg_reproduction requires write_files=False")
    (
        generator_manifest, command_table, correction_matrix, phase_matrix,
        command_table_sha, matrix_sha, generator_config_sha, generator_source_sha,
        transport_observation, transport_audit_sha, _generator_source_bytes,
    ) = _build_recipe(reproduction_input)
    diagnostic = _preview_diagnostic(
        reproduction_input, command_table, correction_matrix, phase_matrix, status="PREVIEW_ONLY",
    )
    diagnostic_sha = canonical_sha256(diagnostic)
    return TrackBAWGReproductionReceiptV1(
        input_sha=reproduction_input.sha256(),
        generator_module=generator_manifest.module_name,
        generator_source_sha=generator_source_sha,
        generator_config_sha=generator_config_sha,
        runtime_sha=_runtime_identity(),
        command_table_sha=command_table_sha,
        matrix_sha=matrix_sha,
        waveform_bundle_sha=None,
        diagnostic_bundle_sha=diagnostic_sha,
        fourier_report_sha=canonical_sha256(diagnostic["fourier_guard"]),
        output_root=None,
        relative_waveform_paths=(),
        waveform_file_manifest_sha=None,
        write_files=False,
        proposal_sha=None if reproduction_input.proposal is None else reproduction_input.proposal.sha256(),
        eligibility_receipt_sha=(
            None if reproduction_input.eligibility_receipt is None
            else reproduction_input.eligibility_receipt.sha256()
        ),
        rf_command_delta_sha=(
            None if reproduction_input.rf_command_delta is None
            else reproduction_input.rf_command_delta.sha256()
        ),
        rf_mapping_core_sha=(
            None if reproduction_input.rf_mapping_core is None
            else reproduction_input.rf_mapping_core.sha256()
        ),
        coordinate_recipe_sha=(
            None if reproduction_input.coordinate_recipe is None
            else reproduction_input.coordinate_recipe.sha256()
        ),
        transport_dependency_audit_sha=transport_audit_sha,
        transport_dependency_status=transport_observation,
    )


def _verify_bound_awg_source_artifacts(
    reproduction_input: TrackBAWGReproductionInputV1,
) -> None:
    """Rehash and parse every externally supplied source artifact at write time."""
    if reproduction_input.proposal is None \
            or reproduction_input.eligibility_receipt is None \
            or reproduction_input.rf_command_delta is None \
            or reproduction_input.rf_mapping_core is None:
        raise ValueError("bound AWG source verification requires the complete proposal chain")
    expected_hashes = dict(reproduction_input.external_artifact_file_shas)
    artifact_paths = dict(reproduction_input.external_artifact_paths)
    if tuple(expected_hashes) != BOUND_AWG_ARTIFACT_NAMES \
            or tuple(artifact_paths) != BOUND_AWG_ARTIFACT_NAMES:
        raise ValueError("bound AWG source verification requires complete ordered artifact files")
    artifact_bytes: dict[str, bytes] = {}
    for name in BOUND_AWG_ARTIFACT_NAMES:
        path = Path(artifact_paths[name])
        if not path.is_file():
            raise ValueError("bound AWG source artifact changed after input construction")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_hashes[name]:
            raise ValueError("bound AWG source artifact changed after input construction")
        artifact_bytes[name] = payload
    proposal = ResearchRFActionProposalV1.from_canonical_json_bytes(artifact_bytes["proposal"])
    receipt = TrackBProposalEligibilityReceiptV1.from_canonical_json_bytes(
        artifact_bytes["eligibility_receipt"]
    )
    rf_delta = RFCommandDeltaV1.from_canonical_json_bytes(artifact_bytes["rf_delta"])
    rf_core = RegisteredRFMappingCoreV1.from_canonical_json_bytes(artifact_bytes["rf_core"])
    rf_mapping = RegisteredRFToAWGMappingV1.from_canonical_json_bytes(
        artifact_bytes["rf_to_awg_mapping"]
    )
    if proposal.sha256() != reproduction_input.proposal.sha256() \
            or receipt.sha256() != reproduction_input.eligibility_receipt.sha256() \
            or rf_delta.sha256() != reproduction_input.rf_command_delta.sha256() \
            or rf_core.sha256() != reproduction_input.rf_mapping_core.sha256() \
            or rf_core.rf_to_awg_mapping is None \
            or rf_mapping.sha256() != rf_core.rf_to_awg_mapping.sha256():
        raise ValueError("bound AWG source artifacts do not replay the in-memory proposal chain")
    _assert_proposal_receipt_lineage(proposal, receipt)
    source_mapping = rf_core.rf_to_awg_mapping
    if expected_hashes["command_table_source"] != source_mapping.command_table_file_sha \
            or expected_hashes["channel_mapping_source"] != source_mapping.channel_mapping_file_sha \
            or expected_hashes["current_control_snapshot"] != source_mapping.current_control_snapshot_sha:
        raise ValueError("bound AWG source files do not match the RF-to-AWG mapping identities")
    command_source = TrackBAWGCommandTableSourceV2.from_canonical_json_bytes(
        artifact_bytes["command_table_source"],
    )
    channel_source = TrackBAWGChannelMappingSourceV1.from_canonical_json_bytes(
        artifact_bytes["channel_mapping_source"],
    )
    control_snapshot = TrackBAWGCurrentControlSnapshotV1.from_canonical_json_bytes(
        artifact_bytes["current_control_snapshot"],
    )
    expected_command_source = {
        "system_type": source_mapping.system_type,
        "command_table_sha": source_mapping.command_table_sha,
        "frequency_steps": source_mapping.frequency_steps,
        "generator_source_sha": source_mapping.generator_source_sha,
        "current_control_snapshot_sha": source_mapping.current_control_snapshot_sha,
    }
    if any(command_source.to_dict()[name] != value for name, value in expected_command_source.items()):
        raise ValueError("command-table source payload does not match the RF-to-AWG mapping")
    if hashlib.sha256(command_source.raw_command_table_json_bytes).hexdigest() \
            != source_mapping.command_table_sha:
        raise ValueError("command-table source raw bytes do not replay the RF-to-AWG command-table identity")
    expected_channel_source = {
        "system_type": source_mapping.system_type,
        "command_table_sha": source_mapping.command_table_sha,
        "frequency_steps": source_mapping.frequency_steps,
        "current_control_snapshot_sha": source_mapping.current_control_snapshot_sha,
        "bindings": [binding.to_dict() for binding in source_mapping.bindings],
    }
    if any(channel_source.to_dict()[name] != value for name, value in expected_channel_source.items()):
        raise ValueError("channel-mapping source payload does not match the RF-to-AWG mapping")
    if control_snapshot.system_type != rf_core.system_type \
            or control_snapshot.current_amplitude != tuple(rf_core.current_amplitude) \
            or control_snapshot.current_phase_rad != tuple(rf_core.current_phase_rad):
        raise ValueError("current-control source payload does not match the RF mapping core")


def _write_trackb_awg_reproduction_to_root(
    reproduction_input: TrackBAWGReproductionInputV1,
) -> TrackBAWGReproductionReceiptV1:
    """Materialize one already-admitted reproduction into an empty staging root."""
    if not reproduction_input.write_files or reproduction_input.output_root is None:
        raise ValueError("write_trackb_awg_reproduction requires an explicit write-enabled output root")
    if reproduction_input.proposal is None:
        raise ValueError(
            "complete AWG reproduction requires an exact proposal/RF eligibility binding"
        )
    _verify_bound_awg_source_artifacts(reproduction_input)
    output_root = Path(reproduction_input.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("AWG reproduction staging root must be empty")
    (
        generator_manifest, command_table, correction_matrix, phase_matrix,
        command_table_sha, matrix_sha, generator_config_sha, generator_source_sha,
        transport_observation, transport_audit_sha, generator_source_bytes,
    ) = _build_recipe(reproduction_input)
    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=False)
    # The legacy generator is imported only inside a fresh subprocess after its
    # exact source and static configuration have already passed the audit.
    _execute_audited_legacy_generator(
        generator_manifest,
        generator_source_bytes=generator_source_bytes,
        output_root=output_root,
        correction_matrix=correction_matrix,
        phase_matrix=phase_matrix,
    )
    waveform_paths = tuple(
        str(path.relative_to(output_root)).replace("\\", "/")
        for path in sorted(output_root.glob("*.txt"))
    )
    expected_count = len(WAVEFORM_CHANNELS) * len(WAVEFORM_VARIANTS)
    if len(waveform_paths) != expected_count:
        raise ValueError(
            f"AWG reproduction expected {expected_count} waveform text files, found {len(waveform_paths)}"
        )
    diagnostic = _analyze_generated_waveforms(reproduction_input, command_table, output_root)
    report_bytes = canonical_json_bytes({
        "schema": "TrackBAWGReproductionReportV1",
        "input_sha": reproduction_input.sha256(),
        "command_table_sha": command_table_sha,
        "matrix_sha": matrix_sha,
        "generator_config_sha": generator_config_sha,
        "generator_source_sha": generator_source_sha,
        "runtime_sha": _runtime_identity(),
        "proposal_sha": reproduction_input.proposal.sha256(),
        "eligibility_receipt_sha": reproduction_input.eligibility_receipt.sha256(),
        "rf_command_delta_sha": reproduction_input.rf_command_delta.sha256(),
        "rf_mapping_core_sha": reproduction_input.rf_mapping_core.sha256(),
        "external_artifact_file_shas": [list(item) for item in reproduction_input.external_artifact_file_shas],
        "coordinate_recipe_sha": reproduction_input.coordinate_recipe.sha256(),
        "coordinate_recipe": reproduction_input.coordinate_recipe.to_dict(),
        "transport_dependency_audit_sha": transport_audit_sha,
        "transport_dependency_status": transport_observation,
        "waveform_file_manifest": [
            {"path": path, "size": size, "sha256": sha}
            for path, size, sha in _file_manifest(output_root, waveform_paths)
        ],
        "diagnostic": diagnostic,
    })
    (output_root / "trackb_awg_reproduction_report.json").write_bytes(report_bytes)
    waveform_bundle_sha = _bundle_sha(output_root, waveform_paths)
    waveform_file_manifest_sha = canonical_sha256({
        "schema": "TrackBAWGWaveformFileManifestV1",
        "files": [
            {"path": path, "size": size, "sha256": sha}
            for path, size, sha in _file_manifest(output_root, waveform_paths)
        ],
    })
    diagnostic_sha = hashlib.sha256(report_bytes).hexdigest()
    output_paths = tuple(
        str(path.relative_to(output_root)).replace("\\", "/")
        for path in sorted((path for path in output_root.rglob("*") if path.is_file()))
    )
    output_file_manifest_sha = canonical_sha256({
        "schema": "TrackBAWGOutputFileManifestV1",
        "files": [
            {"path": path, "size": size, "sha256": sha}
            for path, size, sha in _file_manifest(output_root, output_paths)
        ],
    })
    return TrackBAWGReproductionReceiptV1(
        input_sha=reproduction_input.sha256(),
        generator_module=generator_manifest.module_name,
        generator_source_sha=generator_source_sha,
        generator_config_sha=generator_config_sha,
        runtime_sha=_runtime_identity(),
        command_table_sha=command_table_sha,
        matrix_sha=matrix_sha,
        waveform_bundle_sha=waveform_bundle_sha,
        diagnostic_bundle_sha=diagnostic_sha,
        fourier_report_sha=canonical_sha256(diagnostic["fourier_guard"]),
        output_root=str(output_root.resolve()),
        relative_waveform_paths=waveform_paths,
        waveform_file_manifest_sha=waveform_file_manifest_sha,
        relative_output_paths=output_paths,
        output_file_manifest_sha=output_file_manifest_sha,
        write_files=True,
        proposal_sha=reproduction_input.proposal.sha256(),
        eligibility_receipt_sha=reproduction_input.eligibility_receipt.sha256(),
        rf_command_delta_sha=reproduction_input.rf_command_delta.sha256(),
        rf_mapping_core_sha=reproduction_input.rf_mapping_core.sha256(),
        coordinate_recipe_sha=reproduction_input.coordinate_recipe.sha256(),
        transport_dependency_audit_sha=transport_audit_sha,
        transport_dependency_status=transport_observation,
    )


def write_trackb_awg_reproduction(
    reproduction_input: TrackBAWGReproductionInputV1,
) -> TrackBAWGReproductionReceiptV1:
    """Atomically publish a fully verified reproduction or leave no final output."""
    if not reproduction_input.write_files or reproduction_input.output_root is None:
        raise ValueError("write_trackb_awg_reproduction requires an explicit write-enabled output root")
    if reproduction_input.proposal is None:
        raise ValueError(
            "complete AWG reproduction requires an exact proposal/RF eligibility binding"
        )
    _verify_bound_awg_source_artifacts(reproduction_input)
    output_root = Path(reproduction_input.output_root)
    if output_root.exists():
        raise ValueError("AWG reproduction output root must not exist before atomic publication")
    output_parent = output_root.parent.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.stage_", dir=output_parent))
    try:
        staged_input = replace(reproduction_input, output_root=str(staging_root.resolve()))
        staged_receipt = _write_trackb_awg_reproduction_to_root(staged_input)
        os.replace(staging_root, output_root)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return replace(staged_receipt, output_root=str(output_root.resolve()))


def verify_trackb_awg_reproduction_pair(
    first: TrackBAWGReproductionReceiptV1,
    second: TrackBAWGReproductionReceiptV1,
) -> TrackBAWGReproductionPairReceiptV1:
    """Require two independently written receipts to agree on every output identity."""
    for receipt in (first, second):
        if not receipt.write_files or receipt.waveform_bundle_sha is None:
            raise ValueError("reproduction pair verification requires two written receipts")
        if receipt.output_root is None:
            raise ValueError("written reproduction receipt is missing its output root")
        root = Path(receipt.output_root)
        if not root.is_dir():
            raise ValueError("written reproduction output root is missing")
        actual_paths = tuple(
            str(path.relative_to(root)).replace("\\", "/")
            for path in sorted(root.glob("*.txt"))
        )
        if actual_paths != receipt.relative_waveform_paths:
            raise ValueError("written reproduction waveform file list changed after receipt")
        manifest = _file_manifest(root, actual_paths)
        manifest_sha = canonical_sha256({
            "schema": "TrackBAWGWaveformFileManifestV1",
            "files": [
                {"path": path, "size": size, "sha256": sha}
                for path, size, sha in manifest
            ],
        })
        if manifest_sha != receipt.waveform_file_manifest_sha:
            raise ValueError("written reproduction waveform file manifest changed after receipt")
        if _bundle_sha(root, actual_paths) != receipt.waveform_bundle_sha:
            raise ValueError("written reproduction waveform bytes changed after receipt")
        actual_output_paths = tuple(
            str(path.relative_to(root)).replace("\\", "/")
            for path in sorted((path for path in root.rglob("*") if path.is_file()))
        )
        if actual_output_paths != receipt.relative_output_paths:
            raise ValueError("written reproduction output file list changed after receipt")
        output_manifest_sha = canonical_sha256({
            "schema": "TrackBAWGOutputFileManifestV1",
            "files": [
                {"path": path, "size": size, "sha256": sha}
                for path, size, sha in _file_manifest(root, actual_output_paths)
            ],
        })
        if output_manifest_sha != receipt.output_file_manifest_sha:
            raise ValueError("written reproduction full output manifest changed after receipt")
        report_path = root / "trackb_awg_reproduction_report.json"
        if not report_path.is_file() or _sha256_file(report_path) != receipt.diagnostic_bundle_sha:
            raise ValueError("written reproduction diagnostic report changed after receipt")
    if first.output_root == second.output_root:
        raise ValueError("reproduction pair must use two isolated output roots")
    fields = (
        "input_sha", "generator_module", "generator_source_sha", "generator_config_sha",
        "runtime_sha", "command_table_sha", "matrix_sha", "waveform_bundle_sha",
        "diagnostic_bundle_sha", "fourier_report_sha", "relative_waveform_paths",
        "waveform_file_manifest_sha", "proposal_sha", "eligibility_receipt_sha",
        "relative_output_paths", "output_file_manifest_sha",
        "rf_command_delta_sha", "rf_mapping_core_sha", "coordinate_recipe_sha",
        "transport_dependency_audit_sha", "transport_dependency_status",
    )
    mismatches = [name for name in fields if getattr(first, name) != getattr(second, name)]
    if mismatches:
        raise ValueError(f"Track B AWG reproduction pair differs in {mismatches}")
    return TrackBAWGReproductionPairReceiptV1(
        first_receipt_sha=first.sha256(),
        second_receipt_sha=second.sha256(),
        command_table_sha=first.command_table_sha,
        matrix_sha=first.matrix_sha,
        waveform_bundle_sha=first.waveform_bundle_sha,
        waveform_file_manifest_sha=first.waveform_file_manifest_sha,
        diagnostic_bundle_sha=first.diagnostic_bundle_sha,
        fourier_report_sha=first.fourier_report_sha,
    )


__all__ = [
    "RFCoordinateToAWGRecipeV1",
    "TrackBAWGChannelMappingSourceV1",
    "TrackBAWGCommandTableRowV1",
    "TrackBAWGCommandTableSourceV2",
    "TrackBAWGCurrentControlSnapshotV1",
    "TrackBAWGReproductionInputV1",
    "TrackBAWGReproductionPairReceiptV1",
    "TrackBAWGReproductionReceiptV1",
    "preview_trackb_awg_reproduction",
    "verify_trackb_awg_reproduction_pair",
    "write_trackb_awg_reproduction",
]
