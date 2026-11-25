"""Fail-closed Track B adapters for immutable V2.4 ``real_static`` rows.

This module deliberately does *not* define another on-disk real-data format.
The only supported source is :class:`V24RealV2ObservationDataset`, wrapped by
its :class:`V24RealGroupedDataset` guard.  Track B adds admission identity,
development-only outer folds, descriptor availability accounting, and a batch
collator suitable for the masked observation runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # Keep import-time diagnostics usable in minimal non-ML environments.
    import torch
    from torch.utils.data import Dataset

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional-ML environment only.
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]
    TORCH_AVAILABLE = False

from .contracts import (
    canonical_json_bytes,
    canonical_sha256,
    parse_canonical_json_object,
    require_exact_canonical_json_round_trip,
    require_sha256,
)
from .dual_descriptor import DualOperationDescriptorV1
from .rf_command_descriptor import RawRFCommandDescriptorV1
from .row_identity import row_identity_manifest_sha, source_row_content_sha


VALID_SPLIT_ROLES = frozenset({"train", "validation", "calibration", "locked"})
TRACKB_OUTER_ROLES = frozenset({"train", "validation", "calibration"})
_EXPECTED_DEVELOPMENT_ROWS = {"3qubit": 739, "4qubit": 1240}
_NATIVE_OPERATION_METADATA_ORIGIN = "NATIVE_SHARD_METADATA"
_NO_OPERATION_METADATA_SIDECAR_SHA = canonical_sha256({
    "schema": "TrackBNoOperationMetadataSidecarV1",
})


@dataclass(frozen=True)
class RealStaticRowMetadataV1:
    """Only grouping metadata; it never carries count or action labels."""

    sample_id: str
    acquisition_run_id: str
    repeat_group_id: str
    trajectory_id: str
    role: str

    def validate(self) -> None:
        if any(type(getattr(self, name)) is not str or not getattr(self, name) for name in (
            "sample_id", "acquisition_run_id", "repeat_group_id", "trajectory_id",
        )):
            raise ValueError("real-static grouping identities must be nonempty strings")
        if self.role not in VALID_SPLIT_ROLES:
            raise ValueError("unknown real-static split role")


def connected_group_components(rows: Sequence[RealStaticRowMetadataV1]) -> tuple[tuple[str, ...], ...]:
    """Return connected components over acquisition/repeat/trajectory identities."""
    values = tuple(rows)
    if not values or len({row.sample_id for row in values}) != len(values):
        raise ValueError("rows require unique sample IDs")
    for row in values:
        row.validate()
    parent = list(range(len(values)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    seen: dict[tuple[str, str], int] = {}
    for index, row in enumerate(values):
        for kind, identity in (
            ("acquisition_run", row.acquisition_run_id),
            ("repeat_group", row.repeat_group_id),
            ("trajectory", row.trajectory_id),
        ):
            key = (kind, identity)
            if key in seen:
                join(index, seen[key])
            else:
                seen[key] = index
    buckets: dict[int, list[str]] = {}
    for index, row in enumerate(values):
        buckets.setdefault(find(index), []).append(row.sample_id)
    return tuple(tuple(sorted(component)) for component in sorted(buckets.values(), key=lambda item: item[0]))


def validate_group_roles(rows: Sequence[RealStaticRowMetadataV1], *, trainer_opened_roles: Sequence[str]) -> None:
    """Reject components crossing roles and any trainer attempt to open locked rows."""
    values = tuple(rows)
    opened = set(trainer_opened_roles)
    if "locked" in opened:
        raise ValueError("locked rows cannot be opened by the Track B trainer")
    if not opened <= (VALID_SPLIT_ROLES - {"locked"}):
        raise ValueError("trainer requested an unknown or locked role")
    by_id = {row.sample_id: row for row in values}
    for component in connected_group_components(values):
        roles = {by_id[sample_id].role for sample_id in component}
        if len(roles) != 1:
            raise ValueError("connected acquisition/repeat/trajectory groups cannot cross split roles")


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("Track B real-static batching requires PyTorch")


def _sha_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _descriptor_cycle_indices(base: Any) -> tuple[int, ...]:
    """Read registered cycles only from the V2 projector descriptor contract."""

    descriptor = np.asarray(base.operation_descriptors, dtype=np.float64)
    if descriptor.ndim != 2 or descriptor.shape[0] != len(base.operation_ids):
        raise ValueError("V2.4 projector descriptor shape disagrees with operation order")
    if base.system_type == "3qubit":
        return tuple(0 for _ in base.operation_ids)
    encoded = descriptor[:, -3]
    cycles_array = np.rint(encoded * 3.0).astype(np.int64)
    if not np.allclose(encoded, cycles_array / 3.0, atol=1.0e-6, rtol=0.0) \
            or np.any(cycles_array < 0):
        raise ValueError("V2.4 projector descriptor lacks registered cycle metadata")
    return tuple(int(value) for value in cycles_array)


def _source_metadata_from_grouped(
    grouped: Any, *, operation_metadata_sidecar: Any | None = None,
) -> tuple[tuple[str, ...], tuple[int, ...], str, str]:
    """Read only already-admitted V2.4 shard metadata and require agreement.

    The V2.4 loader keeps the projector array in memory but intentionally does
    not expose operation family/cycle fields on each row.  Those fields are
    source-bound shard metadata, so Track B reads and compares them rather than
    inferring them from hidden counts or inventing a second registry.
    """
    from ml_recon.data_v2_4.storage_hdf5 import validate_hdf5_shard

    base = grouped.dataset
    metadata_values = []
    for relative in sorted({str(row["observation_shard"]) for row in base.rows}):
        metadata_values.append(validate_hdf5_shard(base.root / relative)["metadata"])
    if not metadata_values:
        raise ValueError("admitted V2.4 source selected no observation shards")
    per_operation_family_values: set[tuple[str, ...]] = set()
    partial_family_vocabularies: set[tuple[str, ...]] = set()
    native_metadata_missing = False
    for metadata in metadata_values:
        # V2.4 stores the registered family-name vocabulary and its explicit
        # per-operation integer IDs separately.  Older fixtures may already
        # materialize a per-operation name vector; accept only either exact
        # representation, never a guessed family from the operation ID.
        raw_family_names = metadata.get("operation_families", ())
        raw_family_ids = metadata.get("operation_family_ids", ())
        if raw_family_names is None:
            raw_family_names = ()
        if raw_family_ids is None:
            raw_family_ids = ()
        if type(raw_family_names) not in (list, tuple) \
                or type(raw_family_ids) not in (list, tuple):
            raise ValueError("V2.4 operation-family metadata has an invalid container type")
        family_names = tuple(str(value) for value in raw_family_names)
        family_ids = tuple(int(value) for value in raw_family_ids)
        if len(family_names) == len(base.operation_ids):
            families = family_names
        elif len(family_ids) == len(base.operation_ids) and family_names \
                and all(0 <= value < len(family_names) for value in family_ids):
            families = tuple(family_names[value] for value in family_ids)
        else:
            native_metadata_missing = True
            if family_names:
                if any(not value for value in family_names):
                    raise ValueError("V2.4 partial operation-family vocabulary contains an empty name")
                partial_family_vocabularies.add(family_names)
            continue
        if any(not value for value in families):
            raise ValueError("V2.4 operation family names must be nonempty")
        per_operation_family_values.add(families)
    if len(per_operation_family_values) > 1:
        raise ValueError("V2.4 shards disagree on operation-family metadata")
    if len(partial_family_vocabularies) > 1:
        raise ValueError("V2.4 shards disagree on partial operation-family vocabularies")
    native_families = next(iter(per_operation_family_values), None)
    if native_families is not None and native_metadata_missing and operation_metadata_sidecar is None:
        raise ValueError("V2.4 source has incomplete cross-shard operation-family metadata")
    if native_families is not None and not native_metadata_missing:
        families = native_families
        cycles = _descriptor_cycle_indices(base)
        origin = _NATIVE_OPERATION_METADATA_ORIGIN
        if operation_metadata_sidecar is not None:
            if tuple(operation_metadata_sidecar.operation_families) != families \
                    or tuple(operation_metadata_sidecar.cycle_indices) != cycles:
                raise ValueError("detached operation metadata sidecar conflicts with native shard metadata")
            origin = "NATIVE_SHARD_METADATA_VERIFIED_BY_SIDECAR"
    else:
        if operation_metadata_sidecar is None:
            raise ValueError("V2.4 source lacks complete operation-family metadata")
        families = tuple(operation_metadata_sidecar.operation_families)
        cycles = tuple(operation_metadata_sidecar.cycle_indices)
        origin = str(operation_metadata_sidecar.origin)
        partial_vocabulary = next(iter(partial_family_vocabularies), None)
        if partial_vocabulary is not None \
                and partial_vocabulary != tuple(operation_metadata_sidecar.operation_family_vocabulary):
            raise ValueError("partial native family vocabulary conflicts with detached sidecar")
        if len(families) != len(base.operation_ids) or len(cycles) != len(base.operation_ids):
            raise ValueError("detached operation metadata sidecar length disagrees with source order")
    payload = {
        "schema": "TrackBV24OperationMetadataV1",
        "operation_ids": list(base.operation_ids),
        "operation_families": list(families),
        "cycle_indices": list(cycles),
        "operation_descriptor_sha": base.operation_descriptor_sha,
        "command_table_sha": base.command_table_sha,
        "origin": origin,
        "sidecar_file_sha": (
            "" if operation_metadata_sidecar is None else operation_metadata_sidecar.sidecar_file_sha
        ),
        "sidecar_semantic_sha": (
            "" if operation_metadata_sidecar is None else operation_metadata_sidecar.sidecar_semantic_sha
        ),
    }
    return families, cycles, canonical_sha256(payload), origin


def _raw_feature_row(descriptor: RawRFCommandDescriptorV1) -> tuple[np.ndarray, np.ndarray]:
    """Vectorize raw RF values *with* their availability bits."""
    names = (
        "raw_amplitude", "relative_phase_sin", "relative_phase_cos", "frequency_hz",
        "frequency_offset_hz", "rf_voltage_or_dac", "per_channel_power",
        "quadratic_terms", "cross_amplitude_terms", "cross_phase_terms",
    )
    values: list[float] = []
    availability: list[bool] = []
    for name in names:
        for value in getattr(descriptor, name):
            values.append(0.0 if value is None else float(value))
            availability.append(value is not None)
    for name in ("total_power", "maximum_power", "amplitude_imbalance"):
        value = getattr(descriptor, name)
        values.append(0.0 if value is None else float(value))
        availability.append(value is not None)
    values.extend((float(descriptor.cycle_index), float(descriptor.within_cycle_index)))
    availability.extend((True, True))
    return np.asarray(values, dtype=np.float32), np.asarray(availability, dtype=np.bool_)


@dataclass(frozen=True)
class TrackBDualDescriptorAttachmentV1:
    """Optional source-bound raw-RF sidecar for a V2.4 projector descriptor.

    Current V2.4 real-static rows contain ideal-projector descriptors but not
    calibrated absolute RF values.  Raw RF is therefore explicitly unavailable
    unless an independently admitted descriptor set is supplied.
    """

    operation_registry_sha: str
    command_table_sha: str
    projector_descriptor_sha: str
    operation_ids: tuple[str, ...]
    raw_rf_features: np.ndarray | None = None
    raw_rf_feature_mask: np.ndarray | None = None
    raw_rf_command_basis_sha: str | None = None
    raw_rf_source_sha: str | None = None
    raw_rf_status: str = "NOT_AVAILABLE_IN_V24_SOURCE"
    schema: str = field(default="TrackBDualDescriptorAttachmentV1", init=False)

    def __post_init__(self) -> None:
        for name in ("operation_registry_sha", "command_table_sha", "projector_descriptor_sha"):
            require_sha256(getattr(self, name), name)
        if not self.operation_ids or len(set(self.operation_ids)) != len(self.operation_ids):
            raise ValueError("descriptor attachment requires a unique operation order")
        if any(type(value) is not str or not value for value in self.operation_ids):
            raise ValueError("descriptor attachment operation IDs must be nonempty strings")
        available = self.raw_rf_features is not None or self.raw_rf_feature_mask is not None
        if available:
            if self.raw_rf_features is None or self.raw_rf_feature_mask is None:
                raise ValueError("raw RF values and availability mask must arrive together")
            features = np.asarray(self.raw_rf_features, dtype=np.float32)
            mask = np.asarray(self.raw_rf_feature_mask, dtype=np.bool_)
            if features.ndim != 2 or features.shape[0] != len(self.operation_ids) or features.shape[1] <= 0:
                raise ValueError("raw RF feature sidecar requires [operations,features]")
            if mask.shape != features.shape or not np.all(np.isfinite(features)):
                raise ValueError("raw RF features/masks are invalid")
            if np.any(features[~mask] != 0.0):
                raise ValueError("unavailable raw RF values must be masked rather than encoded")
            if self.raw_rf_status != "SOURCE_BOUND_AVAILABLE":
                raise ValueError("present raw RF values require SOURCE_BOUND_AVAILABLE status")
            require_sha256(self.raw_rf_command_basis_sha, "raw_rf_command_basis_sha")
            require_sha256(self.raw_rf_source_sha, "raw_rf_source_sha")
            frozen_features = np.array(features, copy=True)
            frozen_mask = np.array(mask, copy=True)
            frozen_features.setflags(write=False)
            frozen_mask.setflags(write=False)
            object.__setattr__(self, "raw_rf_features", frozen_features)
            object.__setattr__(self, "raw_rf_feature_mask", frozen_mask)
        else:
            if self.raw_rf_status != "NOT_AVAILABLE_IN_V24_SOURCE":
                raise ValueError("absent raw RF sidecar must explicitly remain unavailable")
            if self.raw_rf_command_basis_sha is not None or self.raw_rf_source_sha is not None:
                raise ValueError("unavailable raw RF cannot claim a source identity")

    @classmethod
    def unavailable(
        cls, *, operation_registry_sha: str, command_table_sha: str,
        projector_descriptor_sha: str, operation_ids: Sequence[str],
    ) -> "TrackBDualDescriptorAttachmentV1":
        return cls(
            operation_registry_sha=operation_registry_sha,
            command_table_sha=command_table_sha,
            projector_descriptor_sha=projector_descriptor_sha,
            operation_ids=tuple(operation_ids),
        )

    @classmethod
    def from_dual_descriptors(
        cls, descriptors: Sequence[DualOperationDescriptorV1], *, projector_descriptor_sha: str,
    ) -> "TrackBDualDescriptorAttachmentV1":
        values = tuple(descriptors)
        if not values:
            raise ValueError("dual descriptor attachment requires at least one operation")
        registry = values[0].ideal_projector.operation_registry_sha
        table = values[0].ideal_projector.command_table_sha
        basis = values[0].raw_rf_command.rf_command_basis_sha
        if any(
            value.ideal_projector.operation_registry_sha != registry
            or value.ideal_projector.command_table_sha != table
            or value.raw_rf_command.rf_command_basis_sha != basis
            for value in values
        ):
            raise ValueError("dual descriptor sidecar has inconsistent registry/table/basis identities")
        rows = [_raw_feature_row(value.raw_rf_command) for value in values]
        if len({row[0].shape for row in rows}) != 1:
            raise ValueError("raw RF sidecar must preserve one registered feature layout")
        features = np.stack([row[0] for row in rows])
        masks = np.stack([row[1] for row in rows])
        raw_source_sha = canonical_sha256({
            "schema": "TrackBRawRFDescriptorSourceV1",
            "raw_descriptor_sha": [value.raw_rf_command.sha256() for value in values],
            "operation_ids": [value.ideal_projector.operation_id for value in values],
        })
        return cls(
            operation_registry_sha=registry,
            command_table_sha=table,
            projector_descriptor_sha=projector_descriptor_sha,
            operation_ids=tuple(value.ideal_projector.operation_id for value in values),
            raw_rf_features=features,
            raw_rf_feature_mask=masks,
            raw_rf_command_basis_sha=basis,
            raw_rf_source_sha=raw_source_sha,
            raw_rf_status="SOURCE_BOUND_AVAILABLE",
        )

    @property
    def raw_rf_available(self) -> bool:
        return self.raw_rf_features is not None

    def sha256(self) -> str:
        payload: dict[str, object] = {
            "schema": self.schema,
            "operation_registry_sha": self.operation_registry_sha,
            "command_table_sha": self.command_table_sha,
            "projector_descriptor_sha": self.projector_descriptor_sha,
            "operation_ids": list(self.operation_ids),
            "raw_rf_status": self.raw_rf_status,
            "raw_rf_command_basis_sha": self.raw_rf_command_basis_sha,
            "raw_rf_source_sha": self.raw_rf_source_sha,
        }
        if self.raw_rf_features is not None:
            payload["raw_rf_features_sha"] = _sha_array(self.raw_rf_features)
            payload["raw_rf_feature_mask_sha"] = _sha_array(self.raw_rf_feature_mask)
        return canonical_sha256(payload)


def _trackb_row_identity_values(
    item: Mapping[str, Any], *, operation_ids: Sequence[str], operation_families: Sequence[str],
    cycle_indices: Sequence[int], descriptor_attachment: TrackBDualDescriptorAttachmentV1,
    source_role: str,
) -> dict[str, Any]:
    prepared = torch.cat((
        item["prepared_state_command_coefficients_re"].to(dtype=torch.float32),
        item["prepared_state_command_coefficients_im"].to(dtype=torch.float32),
    ))
    prepared_available = bool(item["prepared_state_mask"].item())
    if not prepared_available:
        prepared = torch.zeros_like(prepared)
    if descriptor_attachment.raw_rf_features is None:
        raw_rf = np.zeros((len(operation_ids), 1), dtype=np.float32)
        raw_rf_mask = np.zeros_like(raw_rf, dtype=np.bool_)
    else:
        raw_rf = np.array(descriptor_attachment.raw_rf_features, copy=True)
        raw_rf_mask = np.array(descriptor_attachment.raw_rf_feature_mask, copy=True)
    return {
        "four_cell_counts": item["four_cell_counts"].to(dtype=torch.int64),
        "operation_mask": item["operation_mask"].to(dtype=torch.bool),
        "projector_descriptor": item["operation_descriptors"].to(dtype=torch.float32),
        "raw_rf_descriptor": raw_rf,
        "raw_rf_feature_mask": raw_rf_mask,
        "prepared_command": prepared,
        "prepared_command_mask": torch.full_like(prepared, prepared_available, dtype=torch.bool),
        "exposure_s": item["exposure_s"].to(dtype=torch.float32).reshape(()),
        "exposure_known_mask": item["exposure_known_mask"].to(dtype=torch.bool).reshape(()),
        "session_index": torch.tensor(0, dtype=torch.long),
        "session_known_mask": torch.tensor(True, dtype=torch.bool),
        "global_multi": item["global_multi"].to(dtype=torch.int64),
        "global_multi_mask": item["global_multi_mask"].to(dtype=torch.bool).reshape(()),
        "global_multi_scope": str(item["global_multi_scope"]),
        "operation_ids": tuple(str(value) for value in operation_ids),
        "operation_families": tuple(str(value) for value in operation_families),
        "cycle_indices": tuple(int(value) for value in cycle_indices),
        "system_type": str(item["system_type"]),
        "prepared_command_semantics": "prepared_state_command_coefficients_not_actual_state_or_feedback_target",
        "raw_rf_status": descriptor_attachment.raw_rf_status,
        "sample_id": str(item["sample_id"]),
        "group_id": "|".join((
            str(item["acquisition_run_id"]), str(item["repeat_group_id"]), str(item["trajectory_id"]),
        )),
        "source_role": source_role,
    }


def _trackb_row_identity_records(
    base: Any, *, operation_ids: Sequence[str], operation_families: Sequence[str],
    cycle_indices: Sequence[int], descriptor_attachment: TrackBDualDescriptorAttachmentV1,
) -> tuple[tuple[str, str, str, str], ...]:
    records = []
    for index in range(len(base)):
        item = base[index]
        values = _trackb_row_identity_values(
            item, operation_ids=operation_ids, operation_families=operation_families,
            cycle_indices=cycle_indices, descriptor_attachment=descriptor_attachment,
            source_role="train",
        )
        records.append((
            values["sample_id"], values["group_id"], source_row_content_sha(values), "train",
        ))
    return tuple(sorted(records))


@dataclass(frozen=True)
class TrackBRealDataAdmissionManifestV1:
    """Immutable identity of the V2.4 real-static input used by Track B."""

    build_id: str
    system_type: str
    source_role: str
    ready_sha: str
    contract_sha: str
    schema_sha: str
    index_sha: str
    split_sha: str
    build_manifest_sha: str
    operation_registry_sha: str
    command_table_sha: str
    operation_descriptor_sha: str
    source_rows: int
    source_group_sha: str
    operation_metadata_sha: str
    descriptor_attachment_sha: str
    raw_rf_status: str
    row_identity_sha: str
    operation_metadata_sidecar_file_sha: str = _NO_OPERATION_METADATA_SIDECAR_SHA
    operation_metadata_sidecar_semantic_sha: str = _NO_OPERATION_METADATA_SIDECAR_SHA
    operation_metadata_origin: str = _NATIVE_OPERATION_METADATA_ORIGIN
    schema: str = field(default="TrackBRealDataAdmissionManifestV1", init=False)

    def __post_init__(self) -> None:
        if self.system_type not in _EXPECTED_DEVELOPMENT_ROWS:
            raise ValueError("Track B admission supports only 3qubit/4qubit")
        if self.source_role != "train":
            raise ValueError("Track B admission may open only V2.4 development train rows")
        if type(self.build_id) is not str or not self.build_id or type(self.source_rows) is not int or self.source_rows <= 0:
            raise ValueError("Track B admission build identity/row count is invalid")
        for name in (
            "ready_sha", "contract_sha", "schema_sha", "index_sha", "split_sha",
            "build_manifest_sha", "operation_registry_sha", "command_table_sha",
            "operation_descriptor_sha", "source_group_sha", "operation_metadata_sha",
            "descriptor_attachment_sha",
            "row_identity_sha", "operation_metadata_sidecar_file_sha",
            "operation_metadata_sidecar_semantic_sha",
        ):
            require_sha256(getattr(self, name), name)
        if self.raw_rf_status not in ("SOURCE_BOUND_AVAILABLE", "NOT_AVAILABLE_IN_V24_SOURCE"):
            raise ValueError("Track B admission raw-RF status is invalid")
        if self.operation_metadata_origin not in (
            _NATIVE_OPERATION_METADATA_ORIGIN,
            "NATIVE_SHARD_METADATA_VERIFIED_BY_SIDECAR",
            "DETACHED_POST_HOC_DETERMINISTIC_DERIVATION",
        ):
            raise ValueError("Track B admission operation metadata origin is invalid")
        detached = self.operation_metadata_origin != _NATIVE_OPERATION_METADATA_ORIGIN
        if detached and (
            self.operation_metadata_sidecar_file_sha == _NO_OPERATION_METADATA_SIDECAR_SHA
            or self.operation_metadata_sidecar_semantic_sha == _NO_OPERATION_METADATA_SIDECAR_SHA
        ):
            raise ValueError("detached operation metadata requires exact sidecar identities")
        if not detached and (
            self.operation_metadata_sidecar_file_sha != _NO_OPERATION_METADATA_SIDECAR_SHA
            or self.operation_metadata_sidecar_semantic_sha != _NO_OPERATION_METADATA_SIDECAR_SHA
        ):
            raise ValueError("native-only operation metadata cannot claim a detached sidecar")

    @classmethod
    def from_grouped_dataset(
        cls, grouped: Any, *, operation_metadata_sha: str,
        descriptor_attachment: TrackBDualDescriptorAttachmentV1,
        operation_families: Sequence[str], cycle_indices: Sequence[int],
        operation_metadata_sidecar: Any | None = None,
        operation_metadata_origin: str = _NATIVE_OPERATION_METADATA_ORIGIN,
        allow_noncanonical_test_fixture: bool = False,
    ) -> "TrackBRealDataAdmissionManifestV1":
        base = _require_v24_grouped_dataset(grouped)
        expected = _EXPECTED_DEVELOPMENT_ROWS[base.system_type]
        if len(base) != expected and not allow_noncanonical_test_fixture:
            raise ValueError(
                f"Track B admitted {base.system_type} source requires exactly {expected} development rows; "
                "synthetic fixtures must opt in explicitly"
            )
        if descriptor_attachment.operation_registry_sha != base.operation_registry_sha \
                or descriptor_attachment.command_table_sha != base.command_table_sha \
                or descriptor_attachment.projector_descriptor_sha != base.operation_descriptor_sha \
                or descriptor_attachment.operation_ids != tuple(base.operation_ids):
            raise ValueError("dual descriptor attachment is not bound to the admitted V2.4 projector order")
        metadata_rows = _metadata_rows_from_base(base)
        validate_group_roles(metadata_rows, trainer_opened_roles=("train",))
        identity = base.identity
        row_identity_records = _trackb_row_identity_records(
            base, operation_ids=base.operation_ids, operation_families=operation_families,
            cycle_indices=cycle_indices, descriptor_attachment=descriptor_attachment,
        )
        group_sha = canonical_sha256({
            "schema": "TrackBRealStaticSourceGroupsV1",
            "components": [list(value) for value in connected_group_components(metadata_rows)],
        })
        return cls(
            build_id=str(identity["build_id"]), system_type=base.system_type,
            source_role=base.role, ready_sha=str(identity["ready_sha"]),
            contract_sha=str(identity["contract_sha"]), schema_sha=str(identity["schema_sha"]),
            index_sha=str(identity["index_sha"]), split_sha=str(identity["split_sha"]),
            build_manifest_sha=str(identity["build_manifest_sha"]),
            operation_registry_sha=base.operation_registry_sha,
            command_table_sha=base.command_table_sha,
            operation_descriptor_sha=base.operation_descriptor_sha,
            source_rows=len(base), source_group_sha=group_sha,
            operation_metadata_sha=operation_metadata_sha,
            descriptor_attachment_sha=descriptor_attachment.sha256(),
            raw_rf_status=descriptor_attachment.raw_rf_status,
            row_identity_sha=row_identity_manifest_sha(row_identity_records),
            operation_metadata_sidecar_file_sha=(
                _NO_OPERATION_METADATA_SIDECAR_SHA
                if operation_metadata_sidecar is None
                else operation_metadata_sidecar.sidecar_file_sha
            ),
            operation_metadata_sidecar_semantic_sha=(
                _NO_OPERATION_METADATA_SIDECAR_SHA
                if operation_metadata_sidecar is None
                else operation_metadata_sidecar.sidecar_semantic_sha
            ),
            operation_metadata_origin=(
                operation_metadata_origin
            ),
        )

    def to_dict(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema": self.schema,
            "build_id": self.build_id,
            "system_type": self.system_type,
            "source_role": self.source_role,
            "ready_sha": self.ready_sha,
            "contract_sha": self.contract_sha,
            "schema_sha": self.schema_sha,
            "index_sha": self.index_sha,
            "split_sha": self.split_sha,
            "build_manifest_sha": self.build_manifest_sha,
            "operation_registry_sha": self.operation_registry_sha,
            "command_table_sha": self.command_table_sha,
            "operation_descriptor_sha": self.operation_descriptor_sha,
            "source_rows": self.source_rows,
            "source_group_sha": self.source_group_sha,
            "operation_metadata_sha": self.operation_metadata_sha,
            "descriptor_attachment_sha": self.descriptor_attachment_sha,
            "raw_rf_status": self.raw_rf_status,
            "row_identity_sha": self.row_identity_sha,
            "operation_metadata_sidecar_file_sha": self.operation_metadata_sidecar_file_sha,
            "operation_metadata_sidecar_semantic_sha": self.operation_metadata_sidecar_semantic_sha,
            "operation_metadata_origin": self.operation_metadata_origin,
        }

    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TrackBRealDataAdmissionManifestV1":
        expected = {
            "schema", "build_id", "system_type", "source_role", "ready_sha", "contract_sha",
            "schema_sha", "index_sha", "split_sha", "build_manifest_sha", "operation_registry_sha",
            "command_table_sha", "operation_descriptor_sha", "source_rows", "source_group_sha",
            "operation_metadata_sha", "descriptor_attachment_sha", "raw_rf_status", "row_identity_sha",
            "operation_metadata_sidecar_file_sha", "operation_metadata_sidecar_semantic_sha",
            "operation_metadata_origin",
        }
        if set(payload) != expected or payload.get("schema") != "TrackBRealDataAdmissionManifestV1":
            raise ValueError("Track B real-data admission manifest schema is invalid")
        return cls(
            build_id=payload["build_id"], system_type=payload["system_type"],
            source_role=payload["source_role"], ready_sha=payload["ready_sha"],
            contract_sha=payload["contract_sha"], schema_sha=payload["schema_sha"],
            index_sha=payload["index_sha"], split_sha=payload["split_sha"],
            build_manifest_sha=payload["build_manifest_sha"],
            operation_registry_sha=payload["operation_registry_sha"],
            command_table_sha=payload["command_table_sha"],
            operation_descriptor_sha=payload["operation_descriptor_sha"],
            source_rows=payload["source_rows"], source_group_sha=payload["source_group_sha"],
            operation_metadata_sha=payload["operation_metadata_sha"],
            descriptor_attachment_sha=payload["descriptor_attachment_sha"],
            raw_rf_status=payload["raw_rf_status"], row_identity_sha=payload["row_identity_sha"],
            operation_metadata_sidecar_file_sha=payload["operation_metadata_sidecar_file_sha"],
            operation_metadata_sidecar_semantic_sha=payload["operation_metadata_sidecar_semantic_sha"],
            operation_metadata_origin=payload["operation_metadata_origin"],
        )

    @classmethod
    def from_canonical_json_bytes(cls, payload: bytes) -> "TrackBRealDataAdmissionManifestV1":
        decoded = parse_canonical_json_object(
            payload,
            schema="TrackBRealDataAdmissionManifestV1",
            exact_fields=frozenset({
                "schema", "build_id", "system_type", "source_role", "ready_sha", "contract_sha",
                "schema_sha", "index_sha", "split_sha", "build_manifest_sha", "operation_registry_sha",
                "command_table_sha", "operation_descriptor_sha", "source_rows", "source_group_sha",
                "operation_metadata_sha", "descriptor_attachment_sha", "raw_rf_status", "row_identity_sha",
                "operation_metadata_sidecar_file_sha", "operation_metadata_sidecar_semantic_sha",
                "operation_metadata_origin",
            }),
        )
        manifest = cls.from_dict(decoded)
        require_exact_canonical_json_round_trip(payload, manifest)
        return manifest


@dataclass(frozen=True)
class TrackBOuterFoldPlanV1:
    """Three mutually exclusive development-only folds over whole connected groups."""

    admission_manifest_sha: str
    seed: int
    train_components: tuple[tuple[str, ...], ...]
    validation_components: tuple[tuple[str, ...], ...]
    calibration_components: tuple[tuple[str, ...], ...]
    schema: str = field(default="TrackBOuterFoldPlanV1", init=False)

    def __post_init__(self) -> None:
        require_sha256(self.admission_manifest_sha, "admission_manifest_sha")
        if type(self.seed) is not int:
            raise TypeError("Track B fold seed must be an integer")
        groups = self.train_components + self.validation_components + self.calibration_components
        if not all(group and tuple(sorted(group)) == group for group in groups):
            raise ValueError("Track B folds require nonempty sorted connected components")
        samples = [sample for group in groups for sample in group]
        if len(samples) != len(set(samples)):
            raise ValueError("Track B outer folds overlap a real-static sample or group")
        if not self.train_components or not self.validation_components or not self.calibration_components:
            raise ValueError("Track B folds require nonempty train/validation/calibration components")

    def components_for_role(self, role: str) -> tuple[tuple[str, ...], ...]:
        if role == "train":
            return self.train_components
        if role == "validation":
            return self.validation_components
        if role == "calibration":
            return self.calibration_components
        raise ValueError("Track B outer role must be train, validation, or calibration")

    def sample_ids_for_role(self, role: str) -> frozenset[str]:
        return frozenset(sample for group in self.components_for_role(role) for sample in group)

    def sha256(self) -> str:
        return canonical_sha256({
            "schema": self.schema,
            "admission_manifest_sha": self.admission_manifest_sha,
            "seed": self.seed,
            "train_components": [list(value) for value in self.train_components],
            "validation_components": [list(value) for value in self.validation_components],
            "calibration_components": [list(value) for value in self.calibration_components],
        })


@dataclass(frozen=True)
class TrackBRealFoldAdmissionV1:
    """Typed selected-fold identity, distinct from full-source admission."""

    admission_manifest_sha: str
    outer_fold_sha: str
    outer_role: str
    selected_sample_ids: tuple[str, ...]
    selected_group_ids: tuple[str, ...]
    selected_row_identity_records: tuple[tuple[str, str, str, str], ...]
    selected_row_identity_sha: str
    session_index_bindings: tuple[tuple[str, int], ...]
    schema: str = field(default="TrackBRealFoldAdmissionV1", init=False)

    def __post_init__(self) -> None:
        for name in (
            "admission_manifest_sha", "outer_fold_sha", "selected_row_identity_sha",
        ):
            require_sha256(getattr(self, name), name)
        if self.outer_role != "train":
            raise ValueError("actionful Track B fold admission may contain only the train role")
        for values, name in (
            (self.selected_sample_ids, "selected_sample_ids"),
            (self.selected_group_ids, "selected_group_ids"),
        ):
            if not values or len(set(values)) != len(values) \
                    or any(type(value) is not str or not value for value in values):
                raise ValueError(f"{name} must be a unique nonempty selected-fold allowlist")
        records = tuple(tuple(record) for record in self.selected_row_identity_records)
        if len(records) != len(self.selected_sample_ids) \
                or any(len(record) != 4 for record in records):
            raise ValueError("selected fold requires one complete row record per sample")
        for sample_id, group_id, content_sha, role in records:
            if type(sample_id) is not str or not sample_id \
                    or type(group_id) is not str or not group_id or role != "train":
                raise ValueError("selected fold row records are invalid")
            require_sha256(content_sha, "selected_fold_row_content_sha")
        if {sample_id for sample_id, _group_id, _content_sha, _role in records} \
                != set(self.selected_sample_ids):
            raise ValueError("selected fold row records differ from the selected sample allowlist")
        if {group_id for _sample_id, group_id, _content_sha, _role in records} \
                != set(self.selected_group_ids):
            raise ValueError("selected fold group allowlist differs from its row records")
        if row_identity_manifest_sha(records) != self.selected_row_identity_sha:
            raise ValueError("selected fold row identity SHA does not match its records")
        if len(self.session_index_bindings) != len(self.selected_sample_ids) \
                or tuple(sample_id for sample_id, _session_index in self.session_index_bindings) \
                != self.selected_sample_ids:
            raise ValueError("selected fold session bindings do not cover the selected sample order")
        if any(type(session_index) is not int or session_index < 0 for _sample_id, session_index in self.session_index_bindings):
            raise ValueError("selected fold session bindings must be nonnegative native integers")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "admission_manifest_sha": self.admission_manifest_sha,
            "outer_fold_sha": self.outer_fold_sha,
            "outer_role": self.outer_role,
            "selected_sample_ids": list(self.selected_sample_ids),
            "selected_group_ids": list(self.selected_group_ids),
            "selected_row_identity_records": [list(record) for record in self.selected_row_identity_records],
            "selected_row_identity_sha": self.selected_row_identity_sha,
            "session_index_bindings": [list(binding) for binding in self.session_index_bindings],
        }

    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TrackBRealFoldAdmissionV1":
        expected = {
            "schema", "admission_manifest_sha", "outer_fold_sha", "outer_role",
            "selected_sample_ids", "selected_group_ids", "selected_row_identity_records",
            "selected_row_identity_sha", "session_index_bindings",
        }
        if set(payload) != expected or payload.get("schema") != "TrackBRealFoldAdmissionV1":
            raise ValueError("Track B selected-fold admission schema is invalid")
        sequence_fields = (
            "selected_sample_ids", "selected_group_ids", "selected_row_identity_records",
            "session_index_bindings",
        )
        if any(type(payload[name]) is not list for name in sequence_fields) \
                or any(type(record) is not list for record in payload["selected_row_identity_records"]) \
                or any(type(binding) is not list for binding in payload["session_index_bindings"]):
            raise TypeError("Track B selected-fold admission sequences must be JSON arrays")
        return cls(
            admission_manifest_sha=payload["admission_manifest_sha"],
            outer_fold_sha=payload["outer_fold_sha"],
            outer_role=payload["outer_role"],
            selected_sample_ids=tuple(payload["selected_sample_ids"]),
            selected_group_ids=tuple(payload["selected_group_ids"]),
            selected_row_identity_records=tuple(
                tuple(record) for record in payload["selected_row_identity_records"]
            ),
            selected_row_identity_sha=payload["selected_row_identity_sha"],
            session_index_bindings=tuple(
                tuple(binding) for binding in payload["session_index_bindings"]
            ),
        )

    @classmethod
    def from_canonical_json_bytes(cls, payload: bytes) -> "TrackBRealFoldAdmissionV1":
        decoded = parse_canonical_json_object(
            payload,
            schema="TrackBRealFoldAdmissionV1",
            exact_fields=frozenset({
                "schema", "admission_manifest_sha", "outer_fold_sha", "outer_role",
                "selected_sample_ids", "selected_group_ids", "selected_row_identity_records",
                "selected_row_identity_sha", "session_index_bindings",
            }),
        )
        admission = cls.from_dict(decoded)
        require_exact_canonical_json_round_trip(payload, admission)
        return admission

    @classmethod
    def from_runtime_identity(cls, identity: Any) -> "TrackBRealFoldAdmissionV1":
        return cls(
            admission_manifest_sha=identity.admission_manifest_sha,
            outer_fold_sha=identity.outer_fold_sha,
            outer_role="train",
            selected_sample_ids=tuple(identity.allowed_sample_ids),
            selected_group_ids=tuple(identity.allowed_group_ids),
            selected_row_identity_records=tuple(identity.allowed_row_identities),
            selected_row_identity_sha=identity.row_identity_sha,
            session_index_bindings=tuple(identity.session_index_bindings),
        )


def _metadata_rows_from_base(base: Any) -> tuple[RealStaticRowMetadataV1, ...]:
    rows = tuple(
        RealStaticRowMetadataV1(
            sample_id=str(row["sample_id"]),
            acquisition_run_id=str(row["acquisition_run_id"]),
            repeat_group_id=str(row["repeat_group_id"]),
            trajectory_id=str(row["trajectory_id"]),
            role="train" if str(row["split"]) in {
                "train", "train_target_family", "probe_train",
            } else "locked",
        )
        for row in base.rows
    )
    if not rows:
        raise ValueError("admitted V2.4 dataset selected zero real-static rows")
    return rows


def _require_v24_grouped_dataset(grouped: Any) -> Any:
    """Require the canonical V2.4 class pair rather than a duck-typed reader."""
    from ml_recon.data_v2_4.loaders import V24RealGroupedDataset, V24RealV2ObservationDataset

    if not isinstance(grouped, V24RealGroupedDataset) or not isinstance(grouped.dataset, V24RealV2ObservationDataset):
        raise TypeError("Track B real-static admission requires V24RealGroupedDataset(V24RealV2ObservationDataset)")
    base = grouped.dataset
    if base.role != "train":
        raise ValueError("Track B never opens V2.4 locked rows for training/folding")
    return base


def build_group_safe_outer_folds(
    grouped: Any, *, admission_manifest: TrackBRealDataAdmissionManifestV1,
    seed: int = 20261001, validation_fraction: float = 0.20,
    calibration_fraction: float = 0.20,
) -> TrackBOuterFoldPlanV1:
    """Partition the admitted development source by full connected components."""
    base = _require_v24_grouped_dataset(grouped)
    if admission_manifest.system_type != base.system_type:
        raise ValueError("Track B fold admission system does not match its V2.4 source")
    if not 0.0 < float(validation_fraction) < 1.0 or not 0.0 < float(calibration_fraction) < 1.0:
        raise ValueError("Track B validation/calibration fractions must lie in (0,1)")
    if float(validation_fraction) + float(calibration_fraction) >= 1.0:
        raise ValueError("Track B validation/calibration fractions leave no train partition")
    rows = _metadata_rows_from_base(base)
    validate_group_roles(rows, trainer_opened_roles=("train",))
    if any(row.role != "train" for row in rows):
        raise ValueError("locked V2.4 source row entered Track B fold planning")
    components = connected_group_components(rows)
    current_group_sha = canonical_sha256({
        "schema": "TrackBRealStaticSourceGroupsV1",
        "components": [list(value) for value in components],
    })
    if current_group_sha != admission_manifest.source_group_sha:
        raise ValueError("Track B V2.4 source grouping changed after admission")
    if len(components) < 3:
        raise ValueError("Track B outer folds require at least three disconnected real groups")
    ordered = tuple(sorted(
        components,
        key=lambda component: canonical_sha256({
            "schema": "TrackBOuterFoldOrderingV1", "seed": int(seed), "component": list(component),
        }),
    ))
    validation_count = max(1, int(round(len(ordered) * float(validation_fraction))))
    calibration_count = max(1, int(round(len(ordered) * float(calibration_fraction))))
    if validation_count + calibration_count >= len(ordered):
        raise ValueError("Track B outer fold fractions leave no complete train group")
    calibration = tuple(ordered[:calibration_count])
    validation = tuple(ordered[calibration_count:calibration_count + validation_count])
    train = tuple(ordered[calibration_count + validation_count:])
    return TrackBOuterFoldPlanV1(
        admission_manifest_sha=admission_manifest.sha256(), seed=int(seed),
        train_components=train, validation_components=validation,
        calibration_components=calibration,
    )


class TrackBRealStaticDatasetV1(Dataset):
    """Track B view over an already-admitted V2.4 development fold."""

    def __init__(
        self, grouped: Any, *, admission_manifest: TrackBRealDataAdmissionManifestV1,
        fold_plan: TrackBOuterFoldPlanV1, outer_role: str,
        descriptor_attachment: TrackBDualDescriptorAttachmentV1 | None = None,
        operation_metadata_sidecar: Any | None = None,
    ) -> None:
        _require_torch()
        base = _require_v24_grouped_dataset(grouped)
        if outer_role not in TRACKB_OUTER_ROLES:
            raise ValueError("Track B dataset outer role must be train, validation, or calibration")
        if fold_plan.admission_manifest_sha != admission_manifest.sha256():
            raise ValueError("Track B fold plan is not bound to this data admission manifest")
        families, cycles, metadata_sha, metadata_origin = _source_metadata_from_grouped(
            grouped, operation_metadata_sidecar=operation_metadata_sidecar,
        )
        if metadata_sha != admission_manifest.operation_metadata_sha:
            raise ValueError("Track B operation metadata changed after admission")
        if metadata_origin != admission_manifest.operation_metadata_origin:
            raise ValueError("Track B operation metadata origin changed after admission")
        attachment = descriptor_attachment or TrackBDualDescriptorAttachmentV1.unavailable(
            operation_registry_sha=base.operation_registry_sha,
            command_table_sha=base.command_table_sha,
            projector_descriptor_sha=base.operation_descriptor_sha,
            operation_ids=base.operation_ids,
        )
        if attachment.sha256() != admission_manifest.descriptor_attachment_sha:
            raise ValueError("Track B descriptor attachment changed after admission")
        row_identity_records = _trackb_row_identity_records(
            base, operation_ids=base.operation_ids, operation_families=families,
            cycle_indices=cycles, descriptor_attachment=attachment,
        )
        if row_identity_manifest_sha(row_identity_records) != admission_manifest.row_identity_sha:
            raise ValueError("Track B row content identities changed after admission")
        selected_ids = fold_plan.sample_ids_for_role(outer_role)
        all_ids = tuple(str(value) for value in base.sample_ids)
        indices = tuple(index for index, sample_id in enumerate(all_ids) if sample_id in selected_ids)
        if not indices or len(indices) != len(selected_ids):
            raise ValueError("Track B outer fold does not exactly select admitted V2.4 rows")
        self.grouped = grouped
        self.admission_manifest = admission_manifest
        self.fold_plan = fold_plan
        self.outer_role = outer_role
        self.descriptor_attachment = attachment
        self.operation_families = families
        self.cycle_indices = cycles
        self.operation_metadata_sha = metadata_sha
        self.operation_metadata_origin = metadata_origin
        self.operation_metadata_sidecar = operation_metadata_sidecar
        self.row_identity_records = {
            sample_id: (group_id, content_sha, role)
            for sample_id, group_id, content_sha, role in row_identity_records
        }
        selected_records = tuple(
            (sample_id, self.row_identity_records[sample_id][0],
             self.row_identity_records[sample_id][1], self.row_identity_records[sample_id][2])
            for sample_id in sorted(selected_ids)
        )
        self.row_identity_sha = row_identity_manifest_sha(selected_records)
        self.indices = indices
        self.operation_ids = tuple(base.operation_ids)
        self.session_index = {
            value: index for index, value in enumerate(sorted({
                str(row["acquisition_run_id"]) for row in base.rows
            }))
        }
        self.raw_rf_feature_dim = (
            int(attachment.raw_rf_features.shape[1]) if attachment.raw_rf_features is not None else 1
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if type(index) is not int or index < 0 or index >= len(self.indices):
            raise IndexError("Track B real-static dataset index is out of range")
        item = self.grouped[self.indices[index]]
        if str(item.get("source_split", "")) not in {"train", "train_target_family", "probe_train"}:
            raise ValueError("locked source row entered Track B dataset")
        if bool(item["real_action_supervision_mask"].item()):
            raise ValueError("Track B real-static row exposed forbidden action supervision")
        operation_mask = item["operation_mask"].to(dtype=torch.bool)
        if operation_mask.shape != (len(self.operation_ids),) or not bool(operation_mask.any()):
            raise ValueError("Track B row has an invalid operation availability mask")
        prepared_available = bool(item["prepared_state_mask"].item())
        prepared = torch.cat((
            item["prepared_state_command_coefficients_re"].to(dtype=torch.float32),
            item["prepared_state_command_coefficients_im"].to(dtype=torch.float32),
        ))
        prepared_mask = torch.full_like(prepared, prepared_available, dtype=torch.bool)
        if not prepared_available:
            prepared = torch.zeros_like(prepared)
        if self.descriptor_attachment.raw_rf_features is None:
            raw_rf = torch.zeros((len(self.operation_ids), self.raw_rf_feature_dim), dtype=torch.float32)
            raw_rf_mask = torch.zeros_like(raw_rf, dtype=torch.bool)
        else:
            raw_rf = torch.from_numpy(np.array(self.descriptor_attachment.raw_rf_features, copy=True))
            raw_rf_mask = torch.from_numpy(np.array(self.descriptor_attachment.raw_rf_feature_mask, copy=True))
        acquisition = str(item["acquisition_run_id"])
        if acquisition not in self.session_index:
            raise ValueError("Track B row session identity is not in its admitted source mapping")
        result = {
            "four_cell_counts": item["four_cell_counts"].to(dtype=torch.int64),
            "operation_mask": operation_mask,
            "projector_descriptor": item["operation_descriptors"].to(dtype=torch.float32),
            "raw_rf_descriptor": raw_rf,
            "raw_rf_feature_mask": raw_rf_mask,
            "prepared_command": prepared,
            "prepared_command_mask": prepared_mask,
            "exposure_s": item["exposure_s"].to(dtype=torch.float32).reshape(()),
            "exposure_known_mask": item["exposure_known_mask"].to(dtype=torch.bool).reshape(()),
            "session_index": torch.tensor(self.session_index[acquisition], dtype=torch.long),
            "session_known_mask": torch.tensor(True, dtype=torch.bool),
            "global_multi": item["global_multi"].to(dtype=torch.int64),
            "global_multi_mask": item["global_multi_mask"].to(dtype=torch.bool).reshape(()),
            "global_multi_scope": str(item["global_multi_scope"]),
            "sample_id": str(item["sample_id"]),
            "group_id": "|".join((
                str(item["acquisition_run_id"]), str(item["repeat_group_id"]), str(item["trajectory_id"]),
            )),
            "operation_ids": self.operation_ids,
            "operation_families": self.operation_families,
            "cycle_indices": self.cycle_indices,
            "system_type": str(item["system_type"]),
            "prepared_command_semantics": str(item["prepared_state_semantics"]),
            "raw_rf_status": self.descriptor_attachment.raw_rf_status,
            # The V2.4 source remains the development/train role even when an
            # outer split selects it for held-out evaluation.  ``outer_role``
            # is deliberately separate so source-content identities do not
            # change when a row is routed to validation or calibration.
            "source_role": "train",
            "outer_role": self.outer_role,
            "admission_manifest_sha": self.admission_manifest.sha256(),
            "outer_fold_sha": self.fold_plan.sha256(),
            "operation_metadata_sidecar_file_sha": self.admission_manifest.operation_metadata_sidecar_file_sha,
            "operation_metadata_sidecar_semantic_sha": self.admission_manifest.operation_metadata_sidecar_semantic_sha,
        }
        sample_id = result["sample_id"]
        expected = self.row_identity_records.get(sample_id)
        if expected is None or expected[0] != result["group_id"]:
            raise ValueError("Track B row identity mapping is not in the admitted source manifest")
        content_sha = source_row_content_sha(result)
        if content_sha != expected[1]:
            raise ValueError("Track B row content changed after admission")
        result["row_content_sha"] = content_sha
        result["row_identity_sha"] = self.row_identity_sha
        if "read_count_scalar" in item:
            from .native_read_trials import NATIVE_READ_ORIGIN

            result["native_read_count_scalar"] = item["read_count_scalar"].to(dtype=torch.int64).reshape(())
            result["native_read_count_origin"] = NATIVE_READ_ORIGIN
        return result

    def close(self) -> None:
        self.grouped.close()


def _collate_trackb_real_static_items(
    items: Sequence[Mapping[str, Any]], *, epoch: int, view_seed: int,
    query_fraction: float, batch_plan_sha: str | None, batch_index: int | None,
    cursor_start: int | None, allowed_outer_roles: frozenset[str],
    plan_schema: str, read_only_evaluation: bool,
    evaluation_outer_role: str | None,
) -> dict[str, Any]:
    """Collate one homogeneous role-scoped fold without changing source rows."""
    _require_torch()
    values = tuple(items)
    if not values:
        raise ValueError("Track B real-static collate received no rows")
    if type(epoch) is not int or type(view_seed) is not int:
        raise TypeError("Track B collate epoch/view_seed must be integers")
    if not np.isfinite(float(query_fraction)) or not 0.0 < float(query_fraction) < 1.0:
        raise ValueError("Track B collate query_fraction must lie in (0,1)")
    if batch_index is not None and (type(batch_index) is not int or batch_index < 0):
        raise ValueError("Track B collate batch_index must be a nonnegative integer when supplied")
    if cursor_start is not None and (type(cursor_start) is not int or cursor_start < 0):
        raise ValueError("Track B collate cursor_start must be a nonnegative integer when supplied")
    operation_ids = tuple(values[0]["operation_ids"])
    families = tuple(values[0]["operation_families"])
    cycles = tuple(values[0]["cycle_indices"])
    system = str(values[0]["system_type"])
    for item in values:
        if tuple(item["operation_ids"]) != operation_ids \
                or tuple(item["operation_families"]) != families \
                or tuple(item["cycle_indices"]) != cycles \
                or str(item["system_type"]) != system:
            raise ValueError("Track B collate cannot mix operation metadata or systems")
        if str(item["prepared_command_semantics"]) != (
            "prepared_state_command_coefficients_not_actual_state_or_feedback_target"
        ):
            raise ValueError("Track B prepared command semantics are not the V2.4 registered semantics")
        admission_shas = {str(item.get("admission_manifest_sha", "")) for item in values}
        fold_shas = {str(item.get("outer_fold_sha", "")) for item in values}
        if len(admission_shas) != 1 or "" in admission_shas:
            raise ValueError("Track B collate requires one bound admission manifest SHA")
        if len(fold_shas) != 1 or "" in fold_shas:
            raise ValueError("Track B collate requires one bound outer-fold SHA")
    source_roles = tuple(str(item.get("source_role", "")) for item in values)
    if any(role != "train" for role in source_roles):
        raise ValueError("Track B collate may consume only development/train source rows")
    outer_roles = tuple(str(item.get("outer_role", item.get("source_role", ""))) for item in values)
    if any(role not in allowed_outer_roles for role in outer_roles):
        raise ValueError("Track B collate outer role is outside its explicitly admitted role set")
    from .observation_plate import deterministic_query_mask

    query_masks = torch.stack([
        torch.from_numpy(deterministic_query_mask(
            sample_id=str(item["sample_id"]), epoch=epoch, view_seed=view_seed,
            operation_ids=operation_ids, operation_families=families,
            cycle_indices=cycles, query_fraction=query_fraction,
        ))
        for item in values
    ]).to(dtype=torch.bool)
    operation_mask = torch.stack([item["operation_mask"] for item in values]).to(dtype=torch.bool)
    if bool(torch.any(query_masks & ~operation_mask)):
        raise ValueError("Track B deterministic query plan selected an unavailable operation")
    visible_operation_mask = operation_mask & ~query_masks
    if not bool(torch.all(visible_operation_mask.any(dim=1))):
        raise ValueError("every Track B row requires at least one visible operation")
    four_cell = torch.stack([item["four_cell_counts"] for item in values]).to(dtype=torch.int64)
    if batch_plan_sha is None:
        plan_payload: dict[str, object] = {
            "schema": plan_schema,
            "sample_ids": [str(item["sample_id"]) for item in values],
            "epoch": epoch,
            "view_seed": view_seed,
            "query_fraction": float(query_fraction),
        }
        if read_only_evaluation:
            plan_payload["outer_role"] = evaluation_outer_role
        resolved_batch_plan_sha = canonical_sha256(plan_payload)
    else:
        resolved_batch_plan_sha = batch_plan_sha
    result: dict[str, Any] = {
        "four_cell_counts": four_cell,
        "visible_four_cell_counts": four_cell * visible_operation_mask.unsqueeze(-1).to(dtype=four_cell.dtype),
        "operation_mask": operation_mask,
        "query_mask": query_masks,
        "visible_operation_mask": visible_operation_mask,
        "projector_descriptor": torch.stack([item["projector_descriptor"] for item in values]),
        "raw_rf_descriptor": torch.stack([item["raw_rf_descriptor"] for item in values]),
        "raw_rf_feature_mask": torch.stack([item["raw_rf_feature_mask"] for item in values]).to(dtype=torch.bool),
        "prepared_command": torch.stack([item["prepared_command"] for item in values]),
        "prepared_command_mask": torch.stack([item["prepared_command_mask"] for item in values]).to(dtype=torch.bool),
        "exposure_s": torch.stack([item["exposure_s"] for item in values]),
        "exposure_known_mask": torch.stack([item["exposure_known_mask"] for item in values]).to(dtype=torch.bool),
        "session_index": torch.stack([item["session_index"] for item in values]).to(dtype=torch.long),
        "session_known_mask": torch.stack([item["session_known_mask"] for item in values]).to(dtype=torch.bool),
        "global_multi": torch.stack([item["global_multi"] for item in values]).to(dtype=torch.int64),
        "global_multi_mask": torch.stack([item["global_multi_mask"] for item in values]).to(dtype=torch.bool),
        "sample_ids": tuple(str(item["sample_id"]) for item in values),
        "group_ids": tuple(str(item["group_id"]) for item in values),
        "global_multi_scope": tuple(str(item["global_multi_scope"]) for item in values),
        "operation_ids": operation_ids,
        "operation_families": families,
        "cycle_indices": cycles,
        "system_type": system,
        "epoch": epoch,
        "view_seed": view_seed,
        "query_fraction": float(query_fraction),
        "query_mask_sha": canonical_sha256({
            "schema": "TrackBDeterministicQueryMaskV1",
            "sample_ids": [str(item["sample_id"]) for item in values],
            "epoch": epoch,
            "view_seed": view_seed,
            "query_fraction": float(query_fraction),
            "masks": query_masks.to(dtype=torch.int8).tolist(),
        }),
        "batch_index": batch_index,
        "cursor_start": cursor_start,
        "cursor_end": None if cursor_start is None else cursor_start + len(values),
        "raw_rf_status": tuple(str(item["raw_rf_status"]) for item in values),
        "prepared_command_semantics": "prepared_state_command_coefficients_not_actual_state_or_feedback_target",
        "source_roles": tuple(str(item["source_role"]) for item in values),
        "outer_roles": outer_roles,
        "admission_manifest_sha": next(iter(admission_shas)),
        "outer_fold_sha": next(iter(fold_shas)),
        "row_content_shas": tuple(str(item["row_content_sha"]) for item in values),
        "row_identity_sha": str(values[0].get("row_identity_sha", "")),
        "batch_plan_sha": resolved_batch_plan_sha,
    }
    if len({str(item.get("row_identity_sha", "")) for item in values}) != 1 \
            or not result["row_identity_sha"]:
        raise ValueError("Track B collate requires one bound row-identity manifest SHA")
    require_sha256(str(result["row_identity_sha"]), "row_identity_sha")
    require_sha256(str(result["batch_plan_sha"]), "batch_plan_sha")
    for item in values:
        require_sha256(str(item.get("row_content_sha", "")), "row_content_sha")
    if result["global_multi"].shape[1:] != (9,):
        raise ValueError("Track B global multi must preserve the native [9] V2.4 layout")
    if any("native_read_count_scalar" in item or "native_read_count_origin" in item for item in values):
        from .native_read_trials import NATIVE_READ_ORIGIN, validate_native_read_conservation

        if not all("native_read_count_scalar" in item and item.get("native_read_count_origin") == NATIVE_READ_ORIGIN for item in values):
            raise ValueError("native READ fields must be complete and origin-bound across the batch")
        result["native_read_count_scalar"] = torch.stack([item["native_read_count_scalar"] for item in values])
        result["native_read_count_origin"] = NATIVE_READ_ORIGIN
        validate_native_read_conservation(result)
    if read_only_evaluation:
        if evaluation_outer_role not in {"validation", "calibration"}:
            raise ValueError("Track B read-only collation requires validation or calibration role")
        result["read_only_evaluation"] = True
        result["evaluation_outer_role"] = evaluation_outer_role
    return result


def trackb_real_static_collate(
    items: Sequence[Mapping[str, Any]], *, epoch: int = 0, view_seed: int = 20261001,
    query_fraction: float = 0.25, batch_plan_sha: str | None = None,
    batch_index: int | None = None, cursor_start: int | None = None,
) -> dict[str, Any]:
    """Collate one train-role Track B fold for the bounded pilot runtime only."""
    return _collate_trackb_real_static_items(
        items, epoch=epoch, view_seed=view_seed, query_fraction=query_fraction,
        batch_plan_sha=batch_plan_sha, batch_index=batch_index, cursor_start=cursor_start,
        allowed_outer_roles=frozenset({"train"}), plan_schema="TrackBBatchPlanV1",
        read_only_evaluation=False, evaluation_outer_role=None,
    )


def trackb_real_static_evaluation_collate(
    items: Sequence[Mapping[str, Any]], *, outer_role: str, epoch: int = 0,
    view_seed: int = 20261001, query_fraction: float = 0.25,
    batch_index: int | None = None, cursor_start: int | None = None,
) -> dict[str, Any]:
    """Collate one held-out fold for no-grad evaluation, never optimizer input."""
    if outer_role not in {"validation", "calibration"}:
        raise ValueError("Track B read-only evaluation role must be validation or calibration")
    return _collate_trackb_real_static_items(
        items, epoch=epoch, view_seed=view_seed, query_fraction=query_fraction,
        batch_plan_sha=None, batch_index=batch_index, cursor_start=cursor_start,
        allowed_outer_roles=frozenset({outer_role}),
        plan_schema="TrackBReadOnlyEvaluationBatchV1",
        read_only_evaluation=True, evaluation_outer_role=outer_role,
    )


def open_trackb_v24_real_static_fold(
    dataset_root: str, *, build_id: str, system_type: str, basis_config_root: str | None,
    outer_role: str, fold_seed: int = 20261001,
    descriptor_attachment: TrackBDualDescriptorAttachmentV1 | None = None,
    expected_operation_registry_sha: str = "", expected_command_table_sha: str = "",
    operation_metadata_sidecar_path: str | None = None,
    expected_operation_metadata_sidecar_sha: str = "",
    allow_noncanonical_test_fixture: bool = False,
) -> TrackBRealStaticDatasetV1:
    """Open the sole supported V2.4 source and return one group-safe Track B fold."""
    root = Path(dataset_root)
    if not root.is_dir() or not (root / "READY.json").is_file():
        raise ValueError("Track B V2.4 source root must contain READY.json before loader materialization")
    if bool(operation_metadata_sidecar_path) != bool(expected_operation_metadata_sidecar_sha):
        raise ValueError("Track B sidecar path and expected SHA must be supplied together")
    if operation_metadata_sidecar_path is not None:
        from .operation_metadata_sidecar import require_detached_sidecar_path
        require_detached_sidecar_path(operation_metadata_sidecar_path, dataset_root)
    from ml_recon.data_v2_4.loaders import V24RealGroupedDataset, V24RealV2ObservationDataset

    source = V24RealV2ObservationDataset(
        dataset_root, build_id=build_id, system_type=system_type, role="train",
        expected_operation_registry_sha=expected_operation_registry_sha,
        expected_command_table_sha=expected_command_table_sha,
        basis_config_root=basis_config_root,
    )
    grouped = V24RealGroupedDataset(source)
    try:
        operation_metadata_sidecar = None
        if operation_metadata_sidecar_path is not None:
            from .operation_metadata_sidecar import load_verified_v240_operation_metadata_sidecar
            operation_metadata_sidecar = load_verified_v240_operation_metadata_sidecar(
                operation_metadata_sidecar_path,
                expected_file_sha=expected_operation_metadata_sidecar_sha,
                dataset_root=dataset_root,
                system_type=system_type,
            )
        _families, _cycles, metadata_sha, metadata_origin = _source_metadata_from_grouped(
            grouped, operation_metadata_sidecar=operation_metadata_sidecar,
        )
        attachment = descriptor_attachment or TrackBDualDescriptorAttachmentV1.unavailable(
            operation_registry_sha=source.operation_registry_sha,
            command_table_sha=source.command_table_sha,
            projector_descriptor_sha=source.operation_descriptor_sha,
            operation_ids=source.operation_ids,
        )
        admission = TrackBRealDataAdmissionManifestV1.from_grouped_dataset(
            grouped, operation_metadata_sha=metadata_sha, descriptor_attachment=attachment,
            operation_families=_families, cycle_indices=_cycles,
            operation_metadata_sidecar=operation_metadata_sidecar,
            operation_metadata_origin=metadata_origin,
            allow_noncanonical_test_fixture=allow_noncanonical_test_fixture,
        )
        fold = build_group_safe_outer_folds(
            grouped, admission_manifest=admission, seed=fold_seed,
        )
        return TrackBRealStaticDatasetV1(
            grouped, admission_manifest=admission, fold_plan=fold, outer_role=outer_role,
            descriptor_attachment=attachment,
            operation_metadata_sidecar=operation_metadata_sidecar,
        )
    except Exception:
        grouped.close()
        raise


__all__ = [
    "TORCH_AVAILABLE", "TRACKB_OUTER_ROLES", "VALID_SPLIT_ROLES",
    "RealStaticRowMetadataV1", "TrackBDualDescriptorAttachmentV1",
    "TrackBOuterFoldPlanV1", "TrackBRealDataAdmissionManifestV1", "TrackBRealFoldAdmissionV1",
    "TrackBRealStaticDatasetV1", "build_group_safe_outer_folds",
    "connected_group_components", "open_trackb_v24_real_static_fold",
    "trackb_real_static_collate", "trackb_real_static_evaluation_collate",
    "validate_group_roles",
]
