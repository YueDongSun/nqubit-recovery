"""Typed, counts-only boundary for the Track B OBS_V5 route.

This namespace has no authority to train a policy, infer a real gain, or open
locked evaluation data.  It deliberately retains unknown repeat information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Sequence
import re

import numpy as np

from ..evidence_file_io import Evidence_file_identity, read_evidence_file

OBS_V5_GATES = (
    "G0_INTEGRITY",
    "G1_PROXY_BACKGROUND",
    "G2_SELECTED_MEASUREMENT",
    "G3_W_SECTOR_RECOVERY",
    "G4_SECTOR_VALUE",
    "G5_REAL_ESM_CSM_SHADOW",
    "G6_MODEL_BASED_FEEDBACK_SHADOW",
)
REAL_V5_ARMS = (
    "BG0_RAW",
    "BG1_V7_DETERMINISTIC_DIAGNOSTIC",
    "BG2_V7_PROXY_LATENT",
    "BG3_V7_PROXY_LATENT_ROBUST",
    "S0_COUNTS_ONLY",
    "S1_SECTOR_ONLY",
    "W0_PURE_HMLE",
    "W1_HIERARCHICAL_W_SECTOR_DETACHED",
    "W2_JOINT_SECTOR_W",
    "W3_AMORTIZED_W_RESIDUAL",
)
GSI_PROXY_SCHEMA = "GSI_BACKGROUND_PROXY_V7_RESEARCH_ONLY"
UNBOUND_REPEATS_UNKNOWN = "UNBOUND_REPEATS_UNKNOWN"
MAX_NATIVE_COUNT = int(np.iinfo(np.int64).max)
MAX_FILE_BACKED_ARTIFACT_BYTES = 4 * 1024 * 1024


def validate_native_int64_scalar(value: object, name: str) -> int:
    if type(value) is not int or value < 0 or value > MAX_NATIVE_COUNT:
        raise ValueError(f"{name} must be a native nonnegative int64 scalar")
    return value


def validated_poisson_total_means(
    lambda_values: Sequence[float], exposure_values: Sequence[float]
) -> tuple[float, ...]:
    try:
        means = tuple(
            float(rate) * float(exposure)
            for rate, exposure in zip(lambda_values, exposure_values, strict=True)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("operation lambda/exposure arrays must align") from error
    if not means or not all(np.isfinite(value) and value > 0.0 for value in means):
        raise ValueError("E_o*lambda_o must be finite and strictly positive")
    return means


def _validated_relative_artifact_path(value: object) -> Path:
    """Accept one portable, unambiguous relative artifact spelling."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("artifact binding requires a nonempty relative path")
    if "\\" in value or ":" in value:
        raise ValueError("artifact binding path uses an ambiguous platform spelling")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact binding path escapes the admitted root")
    if any(
        not part or part in (".", "..") or part.endswith((".", " "))
        for part in relative.parts
    ):
        raise ValueError("artifact binding path has an ambiguous component")
    return relative


_SHA_256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SEMANTIC_ARTIFACT_ROLES = (
    "coordinate_layout",
    "phase_convention",
    "gauge_rule",
    "forward_recipe",
    "jacobian",
    "covariance",
    "whitener_support",
    "nuisance_schema",
    "projector_registry",
    "apparatus",
    "readout",
    "background_prior",
)
REGISTERED_ARTIFACT_ROLES = frozenset(
    {"source", "calibration", "provenance", *SEMANTIC_ARTIFACT_ROLES}
)


@dataclass(frozen=True)
class File_backed_artifact_binding:
    """A path-relative immutable artifact identity for OBS_V5 admission."""

    role: str
    relative_path: str
    sha256: str
    file_identity: Evidence_file_identity | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def validate(self, root: Path) -> Path:
        return self.read_verified_bytes(root)[0]

    def read_verified_bytes(self, root: Path) -> tuple[Path, bytes]:
        """Read once and return the exact bytes whose SHA-256 was verified."""
        if self.role not in REGISTERED_ARTIFACT_ROLES:
            raise ValueError("artifact binding role is not registered")
        if not _SHA_256_PATTERN.fullmatch(self.sha256):
            raise ValueError("artifact binding requires a lowercase SHA-256")
        relative = _validated_relative_artifact_path(self.relative_path)
        root_path = Path(root).absolute()
        candidate = root_path / relative
        canonical_path, contents, file_identity = read_evidence_file(
            candidate,
            f"{self.role} artifact",
            MAX_FILE_BACKED_ARTIFACT_BYTES,
            expected_identity=self.file_identity,
        )
        if hashlib.sha256(contents).hexdigest() != self.sha256:
            raise ValueError("artifact binding content hash mismatch")
        if self.file_identity is None:
            object.__setattr__(self, "file_identity", file_identity)
        return canonical_path, contents


@dataclass(frozen=True)
class Real_v5_file_backed_admission:
    """Rehash source, calibration, and provenance before a CLI evaluation."""

    source: File_backed_artifact_binding
    calibration: File_backed_artifact_binding
    provenance: File_backed_artifact_binding
    semantic_artifacts: tuple[File_backed_artifact_binding, ...] = ()

    def _ordered_bindings(self) -> tuple[File_backed_artifact_binding, ...]:
        primary = (self.source, self.calibration, self.provenance)
        if tuple(binding.role for binding in primary) != (
            "source", "calibration", "provenance",
        ):
            raise ValueError("admission requires source, calibration, and provenance artifacts")
        if tuple(binding.role for binding in self.semantic_artifacts) != SEMANTIC_ARTIFACT_ROLES:
            raise ValueError("admission requires the exact ordered semantic artifact roles")
        return primary + self.semantic_artifacts

    def validate(self, root: Path) -> tuple[Path, ...]:
        bindings = self._ordered_bindings()
        paths = tuple(binding.read_verified_bytes(root)[0] for binding in bindings)
        if len(set(paths)) != len(paths):
            raise ValueError("admission artifacts must be distinct files")
        return paths

    def read_verified_bytes(
        self, root: Path
    ) -> tuple[tuple[Path, bytes], tuple[Path, bytes], tuple[Path, bytes]]:
        values_by_role = self.read_verified_artifacts(root)
        return tuple(
            values_by_role[role] for role in ("source", "calibration", "provenance")
        )

    def read_verified_artifacts(
        self, root: Path
    ) -> dict[str, tuple[Path, bytes]]:
        bindings = self._ordered_bindings()
        values = tuple(binding.read_verified_bytes(root) for binding in bindings)
        if len({path for path, _contents in values}) != len(values):
            raise ValueError("admission artifacts must be distinct files")
        return {
            binding.role: value for binding, value in zip(bindings, values, strict=True)
        }


@dataclass(frozen=True)
class Real_v5_observation_batch:
    system_type: str
    four_cell_counts: np.ndarray
    observed_operation_mask: np.ndarray
    loss_operation_mask: np.ndarray
    acquisition_group_ids: tuple[str, ...]
    global_multi: np.ndarray
    global_multi_scope: tuple[str, ...]
    development_row_mask: np.ndarray
    repeat_status: str = UNBOUND_REPEATS_UNKNOWN
    repeats: np.ndarray | None = None
    gsi_schema: str = GSI_PROXY_SCHEMA
    source_snapshot_sha: str = ""
    manifest_sha: str = ""
    calibration_sha: str = ""
    operation_registry_sha: str = ""
    command_table_sha: str = ""
    development_fold_sha: str = ""
    sample_ids: tuple[str, ...] = ()
    source_timestamps: tuple[str, ...] = ()
    operation_ids: tuple[str, ...] = ()

    def validate(self) -> None:
        counts = np.asarray(self.four_cell_counts)
        if counts.ndim != 3:
            raise ValueError("four-cell counts must be rank-3")
        rows, operation_count, cell_count = counts.shape
        expected_operations = {"3qubit": 75, "4qubit": 256}
        if (
            self.system_type not in expected_operations
            or operation_count != expected_operations[self.system_type]
            or cell_count != 4
        ):
            raise ValueError(
                "real V5 requires native 3Q/4Q [rows,75|256,4] four-cell counts"
            )
        if (
            not np.issubdtype(counts.dtype, np.integer)
            or not np.all(np.isfinite(counts))
            or np.any(counts < 0)
        ):
            raise ValueError("four-cell counts must be finite nonnegative integers")
        if any(int(value) > MAX_NATIVE_COUNT for value in counts.reshape(-1).tolist()):
            raise ValueError("four-cell counts exceed registered native int64 domain")
        for operation_cells in counts.reshape(-1, 4).tolist():
            if sum(int(value) for value in operation_cells) > MAX_NATIVE_COUNT:
                raise ValueError("four-cell total exceeds registered native int64 domain")
        for mask in (self.observed_operation_mask, self.loss_operation_mask):
            array = np.asarray(mask)
            if array.shape != (rows, operation_count) or array.dtype != np.bool_:
                raise ValueError(
                    "observed and loss masks must be strict bool [rows,operations]"
                )
        if np.any(
            np.asarray(self.loss_operation_mask, bool)
            & ~np.asarray(self.observed_operation_mask, bool)
        ):
            raise ValueError("loss entries must be observed entries")
        development = np.asarray(self.development_row_mask)
        if (
            len(self.acquisition_group_ids) != rows
            or len(self.sample_ids) != rows
            or len(self.source_timestamps) != rows
            or len(set(self.sample_ids)) != rows
            or any(
                type(value) is not str or not value.strip()
                for values in (
                    self.acquisition_group_ids,
                    self.sample_ids,
                    self.source_timestamps,
                )
                for value in values
            )
            or development.shape != (rows,)
            or development.dtype != np.bool_
        ):
            raise ValueError(
                "group/sample/timestamp identities and development mask must align to rows"
            )
        multi = np.asarray(self.global_multi)
        if (
            multi.shape != (rows, 9)
            or not np.issubdtype(multi.dtype, np.integer)
            or not np.all(np.isfinite(multi))
            or np.any(multi < 0)
        ):
            raise ValueError(
                "global multi must preserve [multi2,six_pairs,multi3,multi4] conservation"
            )
        for multi_row in multi.tolist():
            values = [int(value) for value in multi_row]
            if any(value > MAX_NATIVE_COUNT for value in values) or sum(values[1:7]) > MAX_NATIVE_COUNT:
                raise ValueError("global multi exceeds registered native int64 domain")
            if sum(values[1:7]) != values[0]:
                raise ValueError("global multi must preserve [multi2,six_pairs,multi3,multi4] conservation")
        if len(self.global_multi_scope) != rows or any(
            scope not in ("global", "state") for scope in self.global_multi_scope
        ):
            raise ValueError("global multi is never operation-local")
        group_roles: dict[str, bool] = {}
        for group_id, is_development in zip(
            self.acquisition_group_ids, development, strict=True
        ):
            previous_role = group_roles.setdefault(str(group_id), bool(is_development))
            if previous_role != bool(is_development):
                raise ValueError(
                    "an acquisition group cannot cross the immutable development boundary"
                )
        if self.gsi_schema != GSI_PROXY_SCHEMA:
            raise ValueError("only the research-only V7 GSI proxy is permitted")
        if self.repeats is not None or self.repeat_status != UNBOUND_REPEATS_UNKNOWN:
            raise ValueError(
                "real V5 cannot invent repeat values when source repeats are unknown"
            )
        identities = (
            self.source_snapshot_sha,
            self.manifest_sha,
            self.calibration_sha,
            self.operation_registry_sha,
            self.command_table_sha,
            self.development_fold_sha,
        )
        if any(not _SHA_256_PATTERN.fullmatch(value) for value in identities):
            raise ValueError(
                "source snapshot, manifest, calibration, registry, command table, and fold require lowercase hexadecimal SHA-256 identities"
            )
        if self.operation_ids and (
            len(self.operation_ids) != operation_count
            or len(set(self.operation_ids)) != operation_count
            or any(type(value) is not str or not value.strip() for value in self.operation_ids)
        ):
            raise ValueError("operation identities must be unique and align to native operations")


def validate_static_loss_contract(loss_names: Sequence[str]) -> None:
    forbidden = {
        "action",
        "gain",
        "jacobian",
        "residual",
        "action_mse",
        "gain_quantiles",
        "causal",
        "transition",
        "control",
        "feedback",
    }
    if forbidden & set(loss_names):
        raise ValueError("real_static has no action, gain, or Jacobian supervision")
