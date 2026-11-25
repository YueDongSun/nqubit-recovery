"""Strict measured-data contract for V8 DLCZ latent-noise research.

The likelihood labels in this module are the hash-bound, counter-unwrapped
``signal_bins``, ``idler_bins`` and ``coin_bins`` arrays.  They are observed
counts, not noise-subtracted labels.  The corresponding ``*_bins_raw`` arrays
are finite-width FPGA register snapshots available only for explicit
counter-unwrapping QA; they can never be selected as NB2 supervision.  The
development-only screen loader leaves those raw payloads unopened by default.

Legacy multi-click counters retain their native global scope.  In particular,
the six detector-pair totals are never expanded into operation-local counts.
All nuisance-identification masks for unavailable real evidence are explicit
zeros so synthetic priors cannot silently become measured labels.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ml_recon.measured_data.datasets import MeasuredManifestDataset, resolve_snapshot_paths
from ml_recon.physics_hybrid_v7.geometry import build_operation_frequency_geometry
from ml_recon.physics_hybrid_v8.aod_optical_detuning import (
    AOD_OPTICAL_DETUNING_FEATURE_NAMES,
    build_nominal_aod_detuning_features,
)

from .contracts import MeasuredAuditWaiverV1, stable_json_sha256
from .feedback_data import (
    V8MeasuredDataBundleV1,
    V8MeasuredRowV1,
    load_v8_measured_data_bundle,
)


DLCZ_MEASURED_DATA_SCHEMA = "physics_hybrid_v8_dlcz_measured_data_v1"
DLCZ_MEASURED_DATA_VERSION = 1
DLCZ_CROSS_FIT_SCHEMA = "physics_hybrid_v8_dlcz_group_cross_fit_v1"
MAX_OPERATIONS = 256
MAX_QUBITS = 4
GLOBAL_PAIR_LABELS = ("12", "13", "14", "23", "24", "34")
PROCESSED_COUNT_NAMES = ("signal_bins", "idler_bins", "coin_bins")
RAW_COUNT_NAMES = ("signal_bins_raw", "idler_bins_raw", "coin_bins_raw")
SYSTEM_OPERATION_COUNTS = {"3qubit": 75, "4qubit": 256}
COUNTER_MODULI = {
    "3qubit": {
        "signal_bins": 1 << 32,
        "idler_bins": 1 << 32,
        "coin_bins": 1 << 32,
    },
    "4qubit": {
        "signal_bins": 1 << 9,
        "idler_bins": 1 << 9,
        "coin_bins": 1 << 6,
    },
}
TRAIN_SPLIT_NAMES = frozenset({"train", "train_target_family", "probe_train"})
VALIDATION_SPLIT_NAMES = frozenset(
    {"validation", "validation_target_family", "probe_validation"}
)
DEVELOPMENT_SPLIT_NAMES = TRAIN_SPLIT_NAMES | VALIDATION_SPLIT_NAMES
FORBIDDEN_SPLIT_TOKENS = ("locked", "holdout", "test", "ood")

REGISTERED_OPERATION_FEATURE_NAMES = (
    "system_is_3qubit",
    "system_is_4qubit",
    "normalized_operation_index",
    "operation_mask",
    "x_frequency",
    "y_frequency",
    "x_y_frequency_interaction",
    "x_frequency_squared",
    "y_frequency_squared",
    "cycle_sin",
    "cycle_cos",
    "basis_amplitude_mean",
    "basis_amplitude_std",
    "basis_amplitude_power",
    "basis_phase_sin_mean",
    "basis_phase_cos_mean",
    *(f"basis_mode_amplitude_{index}" for index in range(MAX_QUBITS)),
    *(f"basis_mode_phase_sin_{index}" for index in range(MAX_QUBITS)),
    *(f"basis_mode_phase_cos_{index}" for index in range(MAX_QUBITS)),
    *(f"basis_mode_mask_{index}" for index in range(MAX_QUBITS)),
    *AOD_OPTICAL_DETUNING_FEATURE_NAMES,
)
REGISTERED_OPERATION_FEATURE_SCHEMA = (
    "physics_hybrid_v8_registered_operation_features_v3"
)
REGISTERED_OPERATION_FEATURE_PROVENANCE = (
    "repository_nominal_basis_registry_with_padded_complex_mode_vector_"
    "and_nominal_aod_detuning_sensitivity_not_snapshot_command"
)
LEGACY_GEOMETRY_FEATURE_NAMES = REGISTERED_OPERATION_FEATURE_NAMES[4:16]
STATIC_PRIOR_PHYSICS_FEATURE_NAMES = (
    "target_born_probability",
    "ideal_fock2_selected_mode_click_probability",
    "ideal_fock3_selected_mode_click_probability",
    "ideal_fock4_selected_mode_click_probability",
)

REAL_NOISE_EVIDENCE_MASK_NAMES = (
    "off_window_evidence_mask",
    "dark_rate_evidence_mask",
    "detector_efficiency_evidence_mask",
    "write_power_evidence_mask",
    "hbt_evidence_mask",
    "operation_local_multi_evidence_mask",
    "clean_count_label_mask",
    "clean_rho_label_mask",
    "action_supervision_mask",
)
PAYLOAD_INTEGRITY_CHECK_MODES = (
    "full_payload_sha256",
    "quick_manifest_entry",
)


@dataclass(frozen=True)
class V8DLCZDevelopmentIndicesV1:
    """Canonical development train/validation membership only."""

    train_indices: np.ndarray
    validation_indices: np.ndarray
    stable_hash: str

    def __post_init__(self) -> None:
        train = _validated_index_array(self.train_indices, "train_indices")
        validation = _validated_index_array(self.validation_indices, "validation_indices")
        if np.intersect1d(train, validation).size:
            raise ValueError("development train and validation indices overlap")
        _require_sha256(self.stable_hash, "stable_hash")


@dataclass(frozen=True)
class V8DLCZCrossFitFoldV1:
    """One group-disjoint cross-fit fold over development rows."""

    fold_index: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_sample_ids_sha256: str
    validation_sample_ids_sha256: str
    group_assignment_sha256: str
    stable_hash: str
    schema: str = DLCZ_CROSS_FIT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DLCZ_CROSS_FIT_SCHEMA:
            raise ValueError("unsupported DLCZ cross-fit schema")
        if int(self.fold_index) < 0:
            raise ValueError("fold_index must be non-negative")
        train = _validated_index_array(self.train_indices, "train_indices")
        validation = _validated_index_array(self.validation_indices, "validation_indices")
        if not train.size or not validation.size:
            raise ValueError("every cross-fit fold requires non-empty train and validation rows")
        if np.intersect1d(train, validation).size:
            raise ValueError("cross-fit train and validation indices overlap")
        for name in (
            "train_sample_ids_sha256",
            "validation_sample_ids_sha256",
            "group_assignment_sha256",
            "stable_hash",
        ):
            _require_sha256(getattr(self, name), name)


@dataclass(frozen=True)
class V8DLCZMeasuredDataV1:
    """A manifest-bound trusted-row payload in one padded, audited index.

    Missing rho remains ``None`` in :attr:`rho`; no zero density matrix is
    fabricated.  Padding in operation and target tensors is identified by an
    exact mask and is never an observed count or target coordinate.  The audit
    states whether the payload is the compatible all-trusted view or the
    selection-before-load development-only view.
    """

    sample_ids: Tuple[str, ...]
    split_names: Tuple[str, ...]
    system_types: Tuple[str, ...]
    repeat_group_ids: Tuple[str, ...]
    time_block_ids: Tuple[str, ...]
    acquisition_run_ids: Tuple[str, ...]
    timestamps: np.ndarray
    signal_counts: np.ndarray
    idler_counts: np.ndarray
    coin_counts: np.ndarray
    operation_mask: np.ndarray
    raw_signal_registers: Optional[np.ndarray]
    raw_idler_registers: Optional[np.ndarray]
    raw_coin_registers: Optional[np.ndarray]
    read_count: np.ndarray
    write_count: np.ndarray
    global_multi2_pairs: np.ndarray
    global_multi_counts: np.ndarray
    global_multi_observation_mask: np.ndarray
    global_multi_scope: Tuple[str, ...]
    exposure_s: np.ndarray
    exposure_known_mask: np.ndarray
    rho: Tuple[Optional[np.ndarray], ...]
    rho_available_mask: np.ndarray
    rho_dimension: np.ndarray
    target_amplitudes: np.ndarray
    target_phases_rad: np.ndarray
    target_mask: np.ndarray
    sample_weight: np.ndarray
    off_window_evidence_mask: np.ndarray
    dark_rate_evidence_mask: np.ndarray
    detector_efficiency_evidence_mask: np.ndarray
    write_power_evidence_mask: np.ndarray
    hbt_evidence_mask: np.ndarray
    operation_local_multi_evidence_mask: np.ndarray
    clean_count_label_mask: np.ndarray
    clean_rho_label_mask: np.ndarray
    action_supervision_mask: np.ndarray
    registered_operation_features_by_system: Mapping[str, np.ndarray]
    registered_operation_feature_names: Tuple[str, ...]
    registered_operation_feature_schema: str
    registered_operation_feature_provenance: str
    geometry_hash_by_system: Mapping[str, str]
    manifest_hash: str
    split_hash: str
    stable_hash: str
    audit: Mapping[str, Any]
    raw_registers_retained: bool
    schema: str = DLCZ_MEASURED_DATA_SCHEMA
    schema_version: int = DLCZ_MEASURED_DATA_VERSION

    def __post_init__(self) -> None:
        if self.schema != DLCZ_MEASURED_DATA_SCHEMA or int(self.schema_version) != 1:
            raise ValueError("unsupported V8 DLCZ measured-data schema/version")
        sample_count = len(self.sample_ids)
        if sample_count <= 0 or len(set(self.sample_ids)) != sample_count:
            raise ValueError("DLCZ measured data requires unique sample IDs")
        tuple_fields = (
            self.split_names,
            self.system_types,
            self.repeat_group_ids,
            self.time_block_ids,
            self.acquisition_run_ids,
            self.global_multi_scope,
            self.rho,
        )
        if any(len(values) != sample_count for values in tuple_fields):
            raise ValueError("DLCZ row metadata lengths disagree")
        if set(self.system_types) != set(SYSTEM_OPERATION_COUNTS):
            raise ValueError("DLCZ measured data must retain both 3Q and 4Q rows")
        if any(scope != "global" for scope in self.global_multi_scope):
            raise ValueError("legacy multi-click observations must retain global scope")
        expected_operation_shape = (sample_count, MAX_OPERATIONS)
        for name in ("signal_counts", "idler_counts", "coin_counts", "operation_mask"):
            if np.asarray(getattr(self, name)).shape != expected_operation_shape:
                raise ValueError(f"{name} must have shape {expected_operation_shape}")
        raw_values = (
            self.raw_signal_registers,
            self.raw_idler_registers,
            self.raw_coin_registers,
        )
        if bool(self.raw_registers_retained) != all(value is not None for value in raw_values):
            raise ValueError("raw-register retention flag disagrees with returned arrays")
        if any(value is not None and np.asarray(value).shape != expected_operation_shape for value in raw_values):
            raise ValueError("raw-register arrays must match padded operation shape")
        vector_names = (
            "timestamps",
            "read_count",
            "write_count",
            "global_multi_observation_mask",
            "exposure_s",
            "exposure_known_mask",
            "rho_available_mask",
            "rho_dimension",
            "sample_weight",
            *REAL_NOISE_EVIDENCE_MASK_NAMES,
        )
        for name in vector_names:
            if np.asarray(getattr(self, name)).shape != (sample_count,):
                raise ValueError(f"{name} must have shape ({sample_count},)")
        if self.global_multi2_pairs.shape != (sample_count, len(GLOBAL_PAIR_LABELS)):
            raise ValueError("global_multi2_pairs must be [sample, 6]")
        if self.global_multi_counts.shape != (sample_count, 3):
            raise ValueError("global_multi_counts must be [sample, multi2/3/4]")
        if self.target_amplitudes.shape != (sample_count, MAX_QUBITS):
            raise ValueError("target_amplitudes must be [sample, 4]")
        if self.target_phases_rad.shape != self.target_amplitudes.shape:
            raise ValueError("target phase/amplitude shapes disagree")
        if self.target_mask.shape != self.target_amplitudes.shape:
            raise ValueError("target mask/amplitude shapes disagree")
        if tuple(self.registered_operation_feature_names) != REGISTERED_OPERATION_FEATURE_NAMES:
            raise ValueError("registered operation feature schema changed")
        if self.registered_operation_feature_schema != REGISTERED_OPERATION_FEATURE_SCHEMA:
            raise ValueError("registered operation feature version changed")
        if self.registered_operation_feature_provenance != REGISTERED_OPERATION_FEATURE_PROVENANCE:
            raise ValueError("registered operation feature provenance changed")
        if set(self.registered_operation_features_by_system) != set(SYSTEM_OPERATION_COUNTS):
            raise ValueError("registered operation features must cover both systems")
        feature_count = len(self.registered_operation_feature_names)
        for system_type, features in self.registered_operation_features_by_system.items():
            if np.asarray(features).shape != (MAX_OPERATIONS, feature_count):
                raise ValueError(f"registered feature shape changed for {system_type}")
        if set(self.geometry_hash_by_system) != set(SYSTEM_OPERATION_COUNTS):
            raise ValueError("geometry hashes must cover both systems")
        for name in ("manifest_hash", "split_hash", "stable_hash"):
            _require_sha256(getattr(self, name), name)
        if np.any(self.signal_counts < 0) or np.any(self.idler_counts < 0) or np.any(self.coin_counts < 0):
            raise ValueError("processed count tensors must be non-negative")
        if not np.all(self.global_multi2_pairs.sum(axis=1) == self.global_multi_counts[:, 0]):
            raise ValueError("global detector-pair totals disagree with multi2")
        if any(np.any(np.asarray(getattr(self, name)) != 0) for name in REAL_NOISE_EVIDENCE_MASK_NAMES):
            raise ValueError("current measured corpus cannot claim unavailable real noise/action labels")

    @property
    def sample_count(self) -> int:
        return len(self.sample_ids)

    def registered_operation_features(
        self,
        indices: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Materialize static-only operation context for selected rows."""

        selected = (
            np.arange(self.sample_count, dtype=np.int64)
            if indices is None
            else _validated_subset_indices(indices, self.sample_count)
        )
        feature_count = len(self.registered_operation_feature_names)
        result = np.zeros((len(selected), MAX_OPERATIONS, feature_count), dtype=np.float32)
        for destination, source_index in enumerate(selected):
            system_type = self.system_types[int(source_index)]
            result[destination] = self.registered_operation_features_by_system[system_type]
        return _read_only(result)


def load_v8_dlcz_measured_data(
    manifest_dir: str | Path = "ml_recon/reports/measured_data_v2_3",
    *,
    data_root: str | Path = "data",
    audit_waiver: Optional[MeasuredAuditWaiverV1 | Mapping[str, Any]] = None,
    curated_corpus_contract: Optional[Mapping[str, Any]] = None,
    include_raw_registers: bool = False,
    now: Optional[datetime] = None,
) -> V8DLCZMeasuredDataV1:
    """Load and audit every trusted row; never filter by development split.

    Raw register files are always hash/range/modulo audited.  The
    ``include_raw_registers`` flag controls only whether those QA arrays remain
    in the returned object, not whether they are validated.
    """

    bundle = load_v8_measured_data_bundle(
        manifest_dir,
        audit_waiver=audit_waiver,
        curated_corpus_contract=curated_corpus_contract,
        now=now,
    )
    return _materialize_v8_dlcz_rows(
        bundle,
        selected_rows=bundle.rows,
        data_root=data_root,
        include_raw_registers=include_raw_registers,
        audit_raw_registers=True,
        payload_scope="all_trusted_rows",
        selection_before_payload_load=False,
    )


def load_v8_dlcz_development_data(
    manifest_dir: str | Path = "ml_recon/reports/measured_data_v2_3",
    *,
    data_root: str | Path = "data",
    audit_waiver: Optional[MeasuredAuditWaiverV1 | Mapping[str, Any]] = None,
    curated_corpus_contract: Optional[Mapping[str, Any]] = None,
    include_raw_registers: bool = False,
    include_rho_payloads: bool = True,
    integrity_check_mode: str = "full_payload_sha256",
    now: Optional[datetime] = None,
) -> V8DLCZMeasuredDataV1:
    """Load only canonical development payloads after a metadata-only audit.

    The whole measured manifest, split assignment and their hashes are parsed
    first, but payload paths are resolved only after selecting the six
    canonical development split names.  Locked final/holdout/test/OOD snapshot
    manifests, processed arrays, raw registers and rho files are therefore
    never opened by this API.  Raw FPGA registers are also zero-open by
    default; callers must explicitly opt into their counter-unwrapping QA.
    """

    bundle = load_v8_measured_data_bundle(
        manifest_dir,
        audit_waiver=audit_waiver,
        curated_corpus_contract=curated_corpus_contract,
        now=now,
    )
    selected_rows = tuple(
        row for row in bundle.rows if row.split_name in DEVELOPMENT_SPLIT_NAMES
    )
    if not selected_rows:
        raise ValueError("development-only DLCZ payload selection is empty")
    selected_names = {row.split_name for row in selected_rows}
    if not selected_names.issubset(DEVELOPMENT_SPLIT_NAMES):
        raise RuntimeError("development-only selection admitted a forbidden split")
    return _materialize_v8_dlcz_rows(
        bundle,
        selected_rows=selected_rows,
        data_root=data_root,
        include_raw_registers=include_raw_registers,
        include_rho_payloads=include_rho_payloads,
        integrity_check_mode=integrity_check_mode,
        audit_raw_registers=bool(include_raw_registers),
        payload_scope="development_only",
        selection_before_payload_load=True,
    )


def _materialize_v8_dlcz_rows(
    bundle: V8MeasuredDataBundleV1,
    *,
    selected_rows: Sequence[V8MeasuredRowV1],
    data_root: str | Path,
    include_raw_registers: bool,
    audit_raw_registers: bool,
    payload_scope: str,
    selection_before_payload_load: bool,
    include_rho_payloads: bool = True,
    integrity_check_mode: str = "full_payload_sha256",
) -> V8DLCZMeasuredDataV1:
    rows = tuple(selected_rows)
    if not rows:
        raise ValueError("DLCZ payload materialization requires at least one row")
    if include_raw_registers and not audit_raw_registers:
        raise ValueError("retaining raw registers requires counter-unwrapping QA")
    if integrity_check_mode not in PAYLOAD_INTEGRITY_CHECK_MODES:
        raise ValueError(
            "integrity_check_mode must be full_payload_sha256 or quick_manifest_entry"
        )
    verify_payload_hash = integrity_check_mode == "full_payload_sha256"
    if payload_scope not in {"all_trusted_rows", "development_only"}:
        raise ValueError("unsupported DLCZ payload scope")
    selected_records = tuple(row.record for row in rows)
    strict_dataset = MeasuredManifestDataset(selected_records, data_root=data_root)
    if tuple(strict_dataset.records) != selected_records:
        raise ValueError("strict manifest dataset ordering disagrees with V8 bundle")
    snapshot_dirs = resolve_snapshot_paths(selected_records, data_root=data_root)
    sample_count = len(rows)
    signal_counts = np.zeros((sample_count, MAX_OPERATIONS), dtype=np.int64)
    idler_counts = np.zeros_like(signal_counts)
    coin_counts = np.zeros_like(signal_counts)
    operation_mask = np.zeros((sample_count, MAX_OPERATIONS), dtype=np.float32)
    raw_signal = np.zeros_like(signal_counts) if include_raw_registers else None
    raw_idler = np.zeros_like(signal_counts) if include_raw_registers else None
    raw_coin = np.zeros_like(signal_counts) if include_raw_registers else None
    read_count = np.zeros(sample_count, dtype=np.int64)
    write_count = np.zeros(sample_count, dtype=np.int64)
    global_pairs = np.zeros((sample_count, len(GLOBAL_PAIR_LABELS)), dtype=np.int64)
    global_multi = np.zeros((sample_count, 3), dtype=np.int64)
    exposure_s = np.zeros(sample_count, dtype=np.float64)
    exposure_mask = np.zeros(sample_count, dtype=np.float32)
    rho_values: List[Optional[np.ndarray]] = []
    rho_mask = np.zeros(sample_count, dtype=np.float32)
    rho_dimension = np.zeros(sample_count, dtype=np.int64)
    target_amplitudes = np.zeros((sample_count, MAX_QUBITS), dtype=np.float64)
    target_phases = np.zeros_like(target_amplitudes)
    target_mask = np.zeros_like(target_amplitudes, dtype=np.float32)
    sample_weight = np.zeros(sample_count, dtype=np.float64)
    raw_unwrap_count = Counter()

    for index, (row, snapshot_dir) in enumerate(zip(rows, snapshot_dirs)):
        record = row.record
        system_type = record.system_type
        operation_count = SYSTEM_OPERATION_COUNTS[system_type]
        operation_mask[index, :operation_count] = 1.0
        processed_hashes = dict(record.count_array_hashes)
        raw_hashes = dict(record.raw_count_array_hashes)
        if set(processed_hashes) != set(PROCESSED_COUNT_NAMES):
            raise ValueError(f"processed count hashes are incomplete for {record.sample_id}")
        if set(raw_hashes) != set(RAW_COUNT_NAMES):
            raise ValueError(f"raw count hashes are incomplete for {record.sample_id}")

        processed_by_name: Dict[str, np.ndarray] = {}
        raw_by_name: Dict[str, np.ndarray] = {}
        for processed_name, raw_name in zip(PROCESSED_COUNT_NAMES, RAW_COUNT_NAMES):
            processed = _load_hash_bound_integer_array(
                snapshot_dir / f"{processed_name}.npy",
                processed_hashes[processed_name],
                expected_shape=(operation_count,),
                field_name=f"{record.sample_id}:{processed_name}",
                verify_payload_hash=verify_payload_hash,
            )
            processed_by_name[processed_name] = processed
            if audit_raw_registers:
                raw = _load_hash_bound_integer_array(
                    snapshot_dir / f"{raw_name}.npy",
                    raw_hashes[raw_name],
                    expected_shape=(operation_count,),
                    field_name=f"{record.sample_id}:{raw_name}",
                    verify_payload_hash=verify_payload_hash,
                )
                modulus = COUNTER_MODULI[system_type][processed_name]
                _validate_counter_unwrap(
                    processed,
                    raw,
                    modulus=modulus,
                    field_name=f"{record.sample_id}:{processed_name}",
                )
                raw_by_name[raw_name] = raw
                raw_unwrap_count[f"{system_type}:{processed_name}"] += int(
                    np.count_nonzero(processed > raw)
                )

        _validate_record_totals(record, processed_by_name)
        signal_counts[index, :operation_count] = processed_by_name["signal_bins"]
        idler_counts[index, :operation_count] = processed_by_name["idler_bins"]
        coin_counts[index, :operation_count] = processed_by_name["coin_bins"]
        if include_raw_registers:
            assert raw_signal is not None and raw_idler is not None and raw_coin is not None
            raw_signal[index, :operation_count] = raw_by_name["signal_bins_raw"]
            raw_idler[index, :operation_count] = raw_by_name["idler_bins_raw"]
            raw_coin[index, :operation_count] = raw_by_name["coin_bins_raw"]

        read_count[index] = _nonnegative_integer_scalar(record.read_count, "read_count")
        write_count[index] = _nonnegative_integer_scalar(record.write_count, "write_count")
        if record.multi_click_scope != "global":
            raise ValueError(f"multi-click scope is not global for {record.sample_id}")
        pair_payload = dict(record.multi2_pairs)
        if set(pair_payload) != set(GLOBAL_PAIR_LABELS):
            raise ValueError(f"global multi2 pair schema changed for {record.sample_id}")
        pair_values = np.asarray(
            [_nonnegative_integer_scalar(pair_payload[name], f"multi2_pairs.{name}") for name in GLOBAL_PAIR_LABELS],
            dtype=np.int64,
        )
        totals = np.asarray(
            [
                _nonnegative_integer_scalar(record.multi2_count, "multi2_count"),
                _nonnegative_integer_scalar(record.multi3_count, "multi3_count"),
                _nonnegative_integer_scalar(record.multi4_count, "multi4_count"),
            ],
            dtype=np.int64,
        )
        if int(pair_values.sum()) != int(totals[0]):
            raise ValueError(f"global multi2 pair sum mismatch for {record.sample_id}")
        global_pairs[index] = pair_values
        global_multi[index] = totals

        if row.exposure_known_mask:
            if abs(float(record.duration_s) - float(record.gate_exposure)) > 1e-9:
                raise ValueError(f"duration/gate exposure mismatch for {record.sample_id}")
            exposure_s[index] = float(row.exposure_s)
            exposure_mask[index] = 1.0
        if include_rho_payloads and row.rho_available_mask:
            rho_path = snapshot_dir / f"rho_{row.rho_kind}.npy"
            density = _load_hash_bound_array(
                rho_path,
                str(row.rho_hash),
                field_name=f"{record.sample_id}:rho_{row.rho_kind}",
                verify_payload_hash=verify_payload_hash,
            )
            expected_dimension = 1 << (3 if system_type == "3qubit" else 4)
            density = _validate_density_matrix(
                density,
                expected_dimension=expected_dimension,
                field_name=record.sample_id,
            )
            rho_values.append(density)
            rho_mask[index] = 1.0
            rho_dimension[index] = expected_dimension
        else:
            rho_values.append(None)

        qubit_count = 3 if system_type == "3qubit" else 4
        amplitude = np.asarray(
            record.exact_target_state_params.get("amplitudes", ()),
            dtype=np.float64,
        )
        phase = np.asarray(
            record.exact_target_state_params.get("phases_rad", ()),
            dtype=np.float64,
        )
        if amplitude.shape != (qubit_count,) or phase.shape != amplitude.shape:
            raise ValueError(f"target state shape changed for {record.sample_id}")
        if not np.all(np.isfinite(amplitude)) or not np.all(np.isfinite(phase)):
            raise ValueError(f"target state contains non-finite values for {record.sample_id}")
        if np.any(amplitude < 0.0) or abs(float(np.sum(np.square(amplitude))) - 1.0) > 1e-6:
            raise ValueError(f"target amplitudes are not a normalized W-state for {record.sample_id}")
        target_amplitudes[index, :qubit_count] = amplitude
        target_phases[index, :qubit_count] = np.angle(np.exp(1j * phase))
        target_mask[index, :qubit_count] = 1.0
        sample_weight[index] = float(record.sample_weight)
        if not np.isfinite(sample_weight[index]) or sample_weight[index] <= 0.0:
            raise ValueError(f"sample weight must be finite and positive for {record.sample_id}")
        if record.command_supervision_mask != 0 or row.real_action_supervision_mask != 0:
            raise ValueError("ordinary measured snapshots cannot become real action supervision")

    registered_features, geometry_hashes = _registered_operation_features()
    evidence_masks = {
        name: np.zeros(sample_count, dtype=np.float32)
        for name in REAL_NOISE_EVIDENCE_MASK_NAMES
    }
    stable_payload = _stable_data_identity_payload(
        bundle,
        geometry_hash_by_system=geometry_hashes,
        registered_operation_features_by_system=registered_features,
    )
    if payload_scope != "all_trusted_rows":
        stable_payload = {
            **stable_payload,
            "payload_scope": str(payload_scope),
            "materialized_sample_ids": [row.sample_id for row in rows],
            "selection_before_payload_load": bool(selection_before_payload_load),
            "raw_register_payloads_opened": bool(audit_raw_registers),
        }
    stable_hash = stable_json_sha256(stable_payload)
    audit = _build_audit(
        bundle,
        stable_hash=stable_hash,
        raw_unwrap_count=raw_unwrap_count,
        include_raw_registers=bool(include_raw_registers),
        include_rho_payloads=bool(include_rho_payloads),
        integrity_check_mode=str(integrity_check_mode),
        geometry_hash_by_system=geometry_hashes,
        registered_operation_features_by_system=registered_features,
        materialized_rows=rows,
        payload_scope=payload_scope,
        selection_before_payload_load=selection_before_payload_load,
        raw_register_payloads_opened=audit_raw_registers,
    )
    return V8DLCZMeasuredDataV1(
        sample_ids=tuple(row.sample_id for row in rows),
        split_names=tuple(row.split_name for row in rows),
        system_types=tuple(row.system_type for row in rows),
        repeat_group_ids=tuple(row.record.repeat_group_id for row in rows),
        time_block_ids=tuple(row.record.time_block_id for row in rows),
        acquisition_run_ids=tuple(row.record.acquisition_run_id for row in rows),
        timestamps=_read_only(np.asarray([row.record.timestamp for row in rows], dtype=np.float64)),
        signal_counts=_read_only(signal_counts),
        idler_counts=_read_only(idler_counts),
        coin_counts=_read_only(coin_counts),
        operation_mask=_read_only(operation_mask),
        raw_signal_registers=_read_only(raw_signal) if raw_signal is not None else None,
        raw_idler_registers=_read_only(raw_idler) if raw_idler is not None else None,
        raw_coin_registers=_read_only(raw_coin) if raw_coin is not None else None,
        read_count=_read_only(read_count),
        write_count=_read_only(write_count),
        global_multi2_pairs=_read_only(global_pairs),
        global_multi_counts=_read_only(global_multi),
        global_multi_observation_mask=_read_only(np.ones(sample_count, dtype=np.float32)),
        global_multi_scope=tuple("global" for _ in rows),
        exposure_s=_read_only(exposure_s),
        exposure_known_mask=_read_only(exposure_mask),
        rho=tuple(rho_values),
        rho_available_mask=_read_only(rho_mask),
        rho_dimension=_read_only(rho_dimension),
        target_amplitudes=_read_only(target_amplitudes),
        target_phases_rad=_read_only(target_phases),
        target_mask=_read_only(target_mask),
        sample_weight=_read_only(sample_weight),
        registered_operation_features_by_system={
            key: _read_only(value) for key, value in registered_features.items()
        },
        registered_operation_feature_names=REGISTERED_OPERATION_FEATURE_NAMES,
        registered_operation_feature_schema=REGISTERED_OPERATION_FEATURE_SCHEMA,
        registered_operation_feature_provenance=REGISTERED_OPERATION_FEATURE_PROVENANCE,
        geometry_hash_by_system=geometry_hashes,
        manifest_hash=bundle.manifest_hash,
        split_hash=bundle.split_hash,
        stable_hash=stable_hash,
        audit=audit,
        raw_registers_retained=bool(include_raw_registers),
        **{name: _read_only(value) for name, value in evidence_masks.items()},
    )


def build_v8_a0_development_contexts(
    data: V8DLCZMeasuredDataV1,
    *,
    manifest_dir: str | Path = "ml_recon/reports/measured_data_v2_3",
    audit_waiver: Optional[MeasuredAuditWaiverV1 | Mapping[str, Any]] = None,
    curated_corpus_contract: Optional[Mapping[str, Any]] = None,
    reference_exposure_s: float = 1200.0,
    now: Optional[datetime] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Rebuild exact V8 A0 development inputs without reopening any payload.

    ``data`` must come from :func:`load_v8_dlcz_development_data` with its
    default zero-open raw-register policy.  This function reloads only the
    manifest/split metadata needed to reproduce the checkpoint-era per-system
    split hashes.  Counts and rho are taken exclusively from the already
    audited development payload, while geometry and retrieval calibration use
    the same repository helpers as V7/V8.
    """

    from system_control.data_ui.rho_fidelity_summary import w_state_indices

    from ml_recon.physics_hybrid_v7.feedback_data import (
        FeedbackMeasuredArraysV1,
        fit_feedback_nuisance_profile,
    )
    from ml_recon.physics_hybrid_v7.physical_ceiling import (
        load_retrieval_rate_vectors,
    )
    from ml_recon.physics_hybrid_v8.train_feedback_models import (
        V8FeedbackSystemContext,
    )

    if not np.isfinite(reference_exposure_s) or float(reference_exposure_s) <= 0.0:
        raise ValueError("reference_exposure_s must be finite and positive")
    data_audit = dict(data.audit)
    required_development_contract = {
        "payload_scope": "development_only",
        "selection_before_payload_load": True,
        "locked_final_payload_open_count": 0,
        "raw_register_payload_open_count": 0,
    }
    mismatches = [
        key
        for key, expected in required_development_contract.items()
        if data_audit.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            "A0 development contexts require the zero-open development loader: "
            f"{mismatches}"
        )
    if not set(data.split_names).issubset(DEVELOPMENT_SPLIT_NAMES):
        raise ValueError("A0 development data contains a forbidden split")

    bundle = load_v8_measured_data_bundle(
        manifest_dir,
        audit_waiver=audit_waiver,
        curated_corpus_contract=curated_corpus_contract,
        now=now,
    )
    if data.manifest_hash != bundle.manifest_hash or data.split_hash != bundle.split_hash:
        raise ValueError("development payload and metadata bundle identities disagree")
    metadata_by_id = {row.sample_id: row for row in bundle.rows}
    if len(metadata_by_id) != len(bundle.rows):
        raise ValueError("metadata bundle sample IDs are not unique")

    checkpoint_split_hash_by_system: Dict[str, str] = {}
    checkpoint_manifest_hash_by_system: Dict[str, str] = {}
    for system_type in SYSTEM_OPERATION_COUNTS:
        digest = hashlib.sha256()
        system_rows = [row for row in bundle.rows if row.system_type == system_type]
        for row in system_rows:
            record = row.record
            split_row = (
                str(row.sample_id),
                str(row.split_name),
                str(record.repeat_group_id),
                str(record.time_block_id),
                format(float(np.float32(record.sample_weight)), ".9g"),
                str(int(row.rho_available_mask > 0)),
                str(int(row.exposure_known_mask > 0)),
            )
            digest.update(("|".join(split_row) + "\n").encode("utf-8"))
        checkpoint_split_hash_by_system[system_type] = digest.hexdigest()
        checkpoint_manifest_hash_by_system[system_type] = bundle.manifest_hash

    contexts: Dict[str, Any] = {}
    development_sample_ids_hash_by_system: Dict[str, str] = {}
    profile_hash_by_system: Dict[str, str] = {}
    for system_type, operation_count in SYSTEM_OPERATION_COUNTS.items():
        selected = np.asarray(
            [
                index
                for index, value in enumerate(data.system_types)
                if value == system_type
            ],
            dtype=np.int64,
        )
        if not selected.size:
            raise ValueError(f"development payload has no {system_type} rows")
        selected_ids = tuple(data.sample_ids[int(index)] for index in selected)
        expected_ids = tuple(
            row.sample_id
            for row in bundle.rows
            if row.system_type == system_type
            and row.split_name in DEVELOPMENT_SPLIT_NAMES
        )
        if selected_ids != expected_ids:
            raise ValueError(
                f"development sample IDs/order differ from metadata for {system_type}"
            )
        development_sample_ids_hash_by_system[system_type] = _sample_ids_sha256(
            selected_ids
        )

        geometry = build_operation_frequency_geometry(system_type)
        if geometry.operation_count != operation_count:
            raise ValueError("repository geometry operation count changed")
        static_features, static_names = geometry.static_features()
        if tuple(static_names) != LEGACY_GEOMETRY_FEATURE_NAMES:
            raise ValueError("repository static operation feature schema changed")
        graph_kernel = geometry.coupling_kernel(
            x_length_mhz=8.0,
            y_length_mhz=8.0,
            same_local_cross_cycle_weight=0.05,
        )
        graph_kernel += np.eye(operation_count, dtype=np.float64)
        graph_kernel /= np.maximum(
            graph_kernel.sum(axis=1, keepdims=True),
            1.0e-12,
        )
        retrieval_vectors, _ = load_retrieval_rate_vectors(
            system_type,
            operation_count,
        )
        retrieval_vectors = np.asarray(retrieval_vectors, dtype=np.float64)
        expected_retrieval_shape = (operation_count, geometry.qubit_count)
        if retrieval_vectors.shape != expected_retrieval_shape:
            raise ValueError("repository retrieval-rate vector shape changed")
        retrieval_vectors /= max(float(np.max(retrieval_vectors)), 1.0e-12)
        retrieval_relative = np.mean(retrieval_vectors, axis=1)
        retrieval_relative /= max(float(np.max(retrieval_relative)), 1.0e-12)

        target_amplitude = np.asarray(
            data.target_amplitudes[selected],
            dtype=np.float64,
        ).copy()
        target_phase = np.asarray(
            data.target_phases_rad[selected],
            dtype=np.float64,
        ).copy()
        target_mask = np.asarray(data.target_mask[selected], dtype=np.float32).copy()
        qubit_count = geometry.qubit_count
        for row_index in range(len(selected)):
            amplitude = np.abs(target_amplitude[row_index, :qubit_count])
            amplitude /= max(float(np.linalg.norm(amplitude)), 1.0e-12)
            target_amplitude[row_index, :qubit_count] = amplitude
            target_phase[row_index, :qubit_count] = np.angle(
                np.exp(1j * target_phase[row_index, :qubit_count])
            )
        target_amplitude[:, qubit_count:] = 0.0
        target_phase[:, qubit_count:] = 0.0

        rho_full: List[Optional[np.ndarray]] = []
        rho_core: List[Optional[np.ndarray]] = []
        rho_available = np.asarray(
            data.rho_available_mask[selected],
            dtype=np.float32,
        ).copy()
        core_indices = w_state_indices(qubit_count)
        for global_index, available in zip(selected, rho_available):
            density_value = data.rho[int(global_index)]
            if available <= 0.5:
                if density_value is not None:
                    raise ValueError("rho mask points to an unexpected density matrix")
                rho_full.append(None)
                rho_core.append(None)
                continue
            if density_value is None:
                raise ValueError("rho availability mask points to missing rho")
            density = np.asarray(density_value, dtype=np.complex128)
            density = 0.5 * (density + density.conj().T)
            density /= max(float(np.real(np.trace(density))), 1.0e-12)
            core = density[np.ix_(core_indices, core_indices)]
            core = 0.5 * (core + core.conj().T)
            core_trace = float(np.real(np.trace(core)))
            if core_trace <= 1.0e-10:
                raise ValueError("development rho core has zero trace")
            core /= core_trace
            rho_full.append(density.astype(np.complex64))
            rho_core.append(core.astype(np.complex64))

        read_count = np.maximum(
            np.asarray(data.read_count[selected], dtype=np.float64),
            1.0,
        )
        multi_upper_rates = (
            np.asarray(data.global_multi_counts[selected], dtype=np.float64)
            / read_count[:, None]
        )
        arrays = FeedbackMeasuredArraysV1(
            system_type=system_type,
            sample_ids=selected_ids,
            split_names=np.asarray(
                [data.split_names[int(index)] for index in selected],
                dtype=str,
            ),
            repeat_group_ids=tuple(
                data.repeat_group_ids[int(index)] for index in selected
            ),
            time_block_ids=tuple(
                data.time_block_ids[int(index)] for index in selected
            ),
            coin_counts=np.asarray(
                data.coin_counts[selected, :operation_count],
                dtype=np.float32,
            ).copy(),
            signal_counts=np.asarray(
                data.signal_counts[selected, :operation_count],
                dtype=np.float32,
            ).copy(),
            idler_counts=np.asarray(
                data.idler_counts[selected, :operation_count],
                dtype=np.float32,
            ).copy(),
            read_count=read_count.astype(np.float32),
            target_amplitude=target_amplitude.astype(np.float32),
            target_phase_rad=target_phase.astype(np.float32),
            target_mask=target_mask,
            rho_full=tuple(rho_full),
            rho_core=tuple(rho_core),
            rho_available_mask=rho_available,
            sample_weight=np.asarray(
                data.sample_weight[selected],
                dtype=np.float32,
            ).copy(),
            multi_upper_rates=multi_upper_rates.astype(np.float32),
            multi_interval_mask=np.ones(len(selected), dtype=np.float32),
            static_operation_features=static_features.astype(np.float32),
            graph_kernel=graph_kernel.astype(np.float32),
            retrieval_rate_vectors=retrieval_vectors.astype(np.float32),
            retrieval_relative_map=retrieval_relative.astype(np.float32),
            geometry_hash=geometry.geometry_hash,
            manifest_hash=bundle.manifest_hash,
        )
        profile = fit_feedback_nuisance_profile(arrays)
        profile_hash_by_system[system_type] = profile.profile_hash

        exposure = np.full(
            len(selected),
            float(reference_exposure_s),
            dtype=np.float32,
        )
        exposure_known = np.zeros(len(selected), dtype=np.float32)
        declared_calibration_hashes: List[Optional[str]] = []
        episode_keys: List[str] = []
        for local_index, sample_id in enumerate(selected_ids):
            row = metadata_by_id[sample_id]
            if row.exposure_known_mask:
                if row.exposure_s is None or float(row.exposure_s) <= 0.0:
                    raise ValueError("known exposure metadata is invalid")
                exposure[local_index] = float(row.exposure_s)
                exposure_known[local_index] = 1.0
            declared_calibration_hashes.append(
                row.calibration_hash if row.calibration_known_mask else None
            )
            episode_key = str(row.paired_episode_group_key or "")
            if episode_key:
                episode_keys.append(episode_key)
        measured_bundle_audit = bundle.audit.to_json_dict()
        measured_bundle_audit["operator_curated_corpus"] = dict(
            bundle.curated_corpus_audit
        )
        contexts[system_type] = V8FeedbackSystemContext(
            arrays=arrays,
            profile=profile,
            geometry=geometry,
            exposure_s=exposure,
            exposure_known_mask=exposure_known,
            declared_calibration_hashes=tuple(declared_calibration_hashes),
            calibration_known_mask=np.zeros(len(selected), dtype=np.float32),
            calibration_identity_match_mask=np.zeros(len(selected), dtype=np.float32),
            paired_episode_group_keys=tuple(sorted(set(episode_keys))),
            reference_exposure_s=float(reference_exposure_s),
            measured_bundle_audit=measured_bundle_audit,
            locked_final_declaration={
                "schema": "v8_dlcz_a0_locked_final_metadata_only_v1",
                "split_names": sorted(
                    {row.split_name for row in bundle.rows}
                    - DEVELOPMENT_SPLIT_NAMES
                ),
                "manifest_hash": bundle.manifest_hash,
                "split_hash": bundle.split_hash,
                "payload_open_count": 0,
            },
            state_pair_transport_set=None,
            state_pair_validation_set=None,
        )

    audit = {
        "schema": "v8_dlcz_a0_development_contexts_v1",
        "development_data_stable_hash": data.stable_hash,
        "development_sample_ids_hash_by_system": (
            development_sample_ids_hash_by_system
        ),
        "context_sample_count_by_system": {
            key: context.arrays.sample_count for key, context in contexts.items()
        },
        "geometry_hash_by_system": {
            key: context.geometry.geometry_hash for key, context in contexts.items()
        },
        "retrieval_rate_vectors_sha256_by_system": {
            key: hashlib.sha256(
                np.ascontiguousarray(context.arrays.retrieval_rate_vectors).tobytes()
            ).hexdigest()
            for key, context in contexts.items()
        },
        "profile_hash_by_system": profile_hash_by_system,
        "checkpoint_split_hash_by_system": checkpoint_split_hash_by_system,
        "checkpoint_manifest_hash_by_system": checkpoint_manifest_hash_by_system,
        "operator_curated_corpus": dict(bundle.curated_corpus_audit),
        "full_manifest_metadata_sample_count": len(bundle.rows),
        "locked_final_metadata_row_count": sum(
            row.split_name not in DEVELOPMENT_SPLIT_NAMES for row in bundle.rows
        ),
        "locked_final_payload_open_count": 0,
        "raw_register_payload_open_count": 0,
        "locked_record_outcome_metadata_parsed": True,
        "strict_blind_final_claim": False,
        "payload_source": "preloaded_v8_dlcz_development_data_only",
        "forbidden_loader_calls": [
            "load_v8_contexts",
            "load_feedback_contexts",
            "load_feedback_measured_arrays",
            "load_operation_point_dataset",
        ],
        "calibration_known_mask_nonzero_count": 0,
        "calibration_identity_match_mask_nonzero_count": 0,
        "state_pair_transport_constructed": False,
    }
    audit["context_binding_hash"] = stable_json_sha256(audit)
    return contexts, audit


def select_development_indices(
    data: V8DLCZMeasuredDataV1,
    split_names: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """Select only predeclared development splits; reject final/holdout/OOD."""

    selected_names = (
        set(DEVELOPMENT_SPLIT_NAMES)
        if split_names is None
        else {str(name) for name in split_names}
    )
    if not selected_names:
        raise ValueError("at least one development split name is required")
    forbidden = sorted(
        name
        for name in selected_names
        if name not in DEVELOPMENT_SPLIT_NAMES
        or any(token in name.lower() for token in FORBIDDEN_SPLIT_TOKENS)
    )
    if forbidden:
        raise ValueError(
            "locked/holdout/test/OOD or unknown splits are forbidden in development selection: "
            + ", ".join(forbidden)
        )
    indices = np.asarray(
        [index for index, name in enumerate(data.split_names) if name in selected_names],
        dtype=np.int64,
    )
    if not indices.size:
        raise ValueError("requested development split selection is empty")
    return _read_only(indices)


def development_train_validation_indices(
    data: V8DLCZMeasuredDataV1,
) -> V8DLCZDevelopmentIndicesV1:
    """Return the immutable canonical train and validation row indices."""

    train = select_development_indices(data, TRAIN_SPLIT_NAMES)
    validation = select_development_indices(data, VALIDATION_SPLIT_NAMES)
    payload = {
        "schema": "physics_hybrid_v8_dlcz_development_indices_v1",
        "data_stable_hash": data.stable_hash,
        "train_sample_ids": [data.sample_ids[int(index)] for index in train],
        "validation_sample_ids": [data.sample_ids[int(index)] for index in validation],
    }
    return V8DLCZDevelopmentIndicesV1(
        train_indices=train,
        validation_indices=validation,
        stable_hash=stable_json_sha256(payload),
    )


def build_group_cross_fit_folds(
    data: V8DLCZMeasuredDataV1,
    indices: Optional[Sequence[int]] = None,
    *,
    n_folds: int = 5,
    seed: int = 20260809,
) -> Tuple[V8DLCZCrossFitFoldV1, ...]:
    """Build deterministic folds over repeat/time connected components.

    Sharing either a repeat group or a time block joins rows into one
    transitive component.  A component is assigned wholly to one validation
    fold, so it can never appear on both sides of a fold.
    """

    fold_count = int(n_folds)
    if fold_count < 2:
        raise ValueError("n_folds must be at least 2")
    selected = (
        select_development_indices(data, TRAIN_SPLIT_NAMES)
        if indices is None
        else _validated_subset_indices(indices, data.sample_count)
    )
    if any(data.split_names[int(index)] not in DEVELOPMENT_SPLIT_NAMES for index in selected):
        raise ValueError("cross-fit rows must come only from development splits")
    components = _group_components(data, selected)
    if len(components) < fold_count:
        raise ValueError("fewer repeat/time components than requested cross-fit folds")

    ranked = sorted(
        components,
        key=lambda component: (
            -len(component),
            stable_json_sha256(
                {
                    "seed": int(seed),
                    "sample_ids": [data.sample_ids[index] for index in component],
                }
            ),
        ),
    )
    fold_components: List[List[Tuple[int, ...]]] = [[] for _ in range(fold_count)]
    fold_sizes = [0] * fold_count
    for component in ranked:
        destination = min(
            range(fold_count),
            key=lambda fold_index: (fold_sizes[fold_index], fold_index),
        )
        fold_components[destination].append(component)
        fold_sizes[destination] += len(component)

    selected_set = set(int(index) for index in selected)
    group_assignment = {
        data.sample_ids[index]: fold_index
        for fold_index, groups in enumerate(fold_components)
        for group in groups
        for index in group
    }
    assignment_hash = stable_json_sha256(
        {
            "schema": DLCZ_CROSS_FIT_SCHEMA,
            "data_stable_hash": data.stable_hash,
            "seed": int(seed),
            "n_folds": fold_count,
            "assignment": group_assignment,
        }
    )
    folds: List[V8DLCZCrossFitFoldV1] = []
    for fold_index, groups in enumerate(fold_components):
        validation_set = {index for group in groups for index in group}
        train_indices = _read_only(np.asarray(sorted(selected_set - validation_set), dtype=np.int64))
        validation_indices = _read_only(np.asarray(sorted(validation_set), dtype=np.int64))
        train_ids = [data.sample_ids[int(index)] for index in train_indices]
        validation_ids = [data.sample_ids[int(index)] for index in validation_indices]
        fold_payload = {
            "schema": DLCZ_CROSS_FIT_SCHEMA,
            "fold_index": fold_index,
            "data_stable_hash": data.stable_hash,
            "group_assignment_sha256": assignment_hash,
            "train_sample_ids": train_ids,
            "validation_sample_ids": validation_ids,
        }
        folds.append(
            V8DLCZCrossFitFoldV1(
                fold_index=fold_index,
                train_indices=train_indices,
                validation_indices=validation_indices,
                train_sample_ids_sha256=_sample_ids_sha256(train_ids),
                validation_sample_ids_sha256=_sample_ids_sha256(validation_ids),
                group_assignment_sha256=assignment_hash,
                stable_hash=stable_json_sha256(fold_payload),
            )
        )
    _validate_fold_group_isolation(data, selected, folds)
    return tuple(folds)


def nb2_count_labels(
    data: V8DLCZMeasuredDataV1,
    *,
    channel: str = "coin",
    source: str = "processed_counter_unwrapped",
) -> np.ndarray:
    """Return observed NB2 labels and reject raw-register supervision."""

    if source != "processed_counter_unwrapped":
        raise ValueError(
            "NB2 labels must use hash-bound processed counter-unwrapped counts; "
            "raw registers are QA-only"
        )
    arrays = {
        "signal": data.signal_counts,
        "idler": data.idler_counts,
        "coin": data.coin_counts,
    }
    if channel not in arrays:
        raise ValueError("channel must be signal, idler, or coin")
    return _read_only(np.asarray(arrays[channel], dtype=np.int64).copy())


def compact_dlcz_data_audit(data: V8DLCZMeasuredDataV1) -> Dict[str, Any]:
    """Return a JSON-safe compact audit for training manifests/reports."""

    return dict(data.audit)


def build_registered_operation_features(
    system_type: str,
    *,
    pad_to_max_operations: bool,
) -> Tuple[np.ndarray, str]:
    """Build the target-free nominal projector descriptor used by DLCZ models.

    The earlier descriptor retained only aggregate basis statistics, so two
    different tomography projectors could be indistinguishable to a model.
    V2 appended the padded complex mode vector as amplitude, phase sine/cosine
    and validity mask.  V3 adds the operator-locked write/read AOD detuning law
    evaluated at the fixed nominal reference point.  These are nominal
    registry/sensitivity facts, not snapshot command evidence and not an
    executed action; the historical operation-point-known mask remains zero.
    """

    if system_type not in SYSTEM_OPERATION_COUNTS:
        raise ValueError("system_type must be 3qubit or 4qubit")
    operation_count = int(SYSTEM_OPERATION_COUNTS[system_type])
    geometry = build_operation_frequency_geometry(system_type)
    static_features, static_names = geometry.static_features()
    if tuple(static_names) != LEGACY_GEOMETRY_FEATURE_NAMES:
        raise ValueError("nominal basis geometry feature schema changed")
    qubit_count = int(geometry.qubit_count)
    amplitude = np.zeros((operation_count, MAX_QUBITS), dtype=np.float64)
    phase_sin = np.zeros_like(amplitude)
    phase_cos = np.zeros_like(amplitude)
    mode_mask = np.zeros_like(amplitude)
    amplitude[:, :qubit_count] = np.asarray(
        geometry.basis_amplitudes, dtype=np.float64
    )
    phases = np.asarray(geometry.basis_phases_rad, dtype=np.float64)
    # The registered tomography POVM uses the repository's fixed phase_sign=-1
    # convention.  Preserve that physical ket, rather than the UI phase table.
    phase_sin[:, :qubit_count] = -np.sin(phases)
    phase_cos[:, :qubit_count] = np.cos(phases)
    mode_mask[:, :qubit_count] = 1.0
    operation_index = np.arange(operation_count, dtype=np.float64)
    normalized_index = operation_index / max(float(operation_count - 1), 1.0)
    prefix = np.stack(
        (
            np.full(operation_count, float(system_type == "3qubit")),
            np.full(operation_count, float(system_type == "4qubit")),
            normalized_index,
            np.ones(operation_count, dtype=np.float64),
        ),
        axis=1,
    )
    aod_features, aod_audit = build_nominal_aod_detuning_features(system_type)
    if (
        aod_features.shape[0] != operation_count
        or aod_audit["geometry_hash"] != geometry.geometry_hash
        or aod_audit["historical_operation_point_authoritative"] is not False
    ):
        raise ValueError("nominal AOD detuning feature binding changed")
    features = np.concatenate(
        (
            prefix,
            static_features,
            amplitude,
            phase_sin,
            phase_cos,
            mode_mask,
            aod_features,
        ),
        axis=1,
    ).astype(np.float32)
    expected_shape = (operation_count, len(REGISTERED_OPERATION_FEATURE_NAMES))
    if features.shape != expected_shape:
        raise AssertionError("registered operation feature width changed")
    if pad_to_max_operations:
        padded = np.zeros(
            (MAX_OPERATIONS, len(REGISTERED_OPERATION_FEATURE_NAMES)),
            dtype=np.float32,
        )
        padded[:operation_count] = features
        features = padded
    return features, geometry.geometry_hash


def _registered_operation_features() -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    features_by_system: Dict[str, np.ndarray] = {}
    geometry_hashes: Dict[str, str] = {}
    for system_type in SYSTEM_OPERATION_COUNTS:
        features, geometry_hash = build_registered_operation_features(
            system_type,
            pad_to_max_operations=True,
        )
        features_by_system[system_type] = features
        geometry_hashes[system_type] = geometry_hash
    return features_by_system, geometry_hashes


def _stable_data_identity_payload(
    bundle: V8MeasuredDataBundleV1,
    *,
    geometry_hash_by_system: Mapping[str, str],
    registered_operation_features_by_system: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    rows = []
    for row in bundle.rows:
        record = row.record
        rows.append(
            {
                "sample_id": row.sample_id,
                "split_name": row.split_name,
                "system_type": row.system_type,
                "snapshot_manifest_hash": record.snapshot_manifest_hash,
                "count_array_hashes": dict(sorted(record.count_array_hashes.items())),
                "raw_count_array_hashes": dict(sorted(record.raw_count_array_hashes.items())),
                "rho_hash": row.rho_hash,
                "exact_target_hash": record.exact_target_hash,
                "global_multi_scope": record.multi_click_scope,
                "global_multi2_pairs": [record.multi2_pairs[name] for name in GLOBAL_PAIR_LABELS],
                "global_multi_counts": [record.multi2_count, record.multi3_count, record.multi4_count],
                "exposure_s": row.exposure_s,
                "exposure_known_mask": row.exposure_known_mask,
            }
        )
    feature_hashes = {
        system_type: hashlib.sha256(
            np.ascontiguousarray(features).view(np.uint8)
        ).hexdigest()
        for system_type, features in registered_operation_features_by_system.items()
    }
    return {
        "schema": DLCZ_MEASURED_DATA_SCHEMA,
        "schema_version": DLCZ_MEASURED_DATA_VERSION,
        "manifest_hash": bundle.manifest_hash,
        "split_hash": bundle.split_hash,
        "summary_sha256": bundle.summary_sha256,
        "rows": rows,
        "likelihood_count_source": "processed_counter_unwrapped_observed_not_noise_subtracted",
        "raw_register_role": "hash_bound_finite_width_counter_qa_only",
        "global_multi_scope": "global_native_observation_never_operation_local",
        "global_pair_labels": list(GLOBAL_PAIR_LABELS),
        "real_noise_evidence_masks": {name: 0 for name in REAL_NOISE_EVIDENCE_MASK_NAMES},
        "registered_operation_feature_names": list(REGISTERED_OPERATION_FEATURE_NAMES),
        "registered_operation_feature_schema": REGISTERED_OPERATION_FEATURE_SCHEMA,
        "registered_operation_feature_provenance": REGISTERED_OPERATION_FEATURE_PROVENANCE,
        "registered_operation_feature_hash_by_system": feature_hashes,
        "geometry_hash_by_system": dict(sorted(geometry_hash_by_system.items())),
        "counter_moduli": COUNTER_MODULI,
    }


def _build_audit(
    bundle: V8MeasuredDataBundleV1,
    *,
    stable_hash: str,
    raw_unwrap_count: Mapping[str, int],
    include_raw_registers: bool,
    include_rho_payloads: bool,
    integrity_check_mode: str,
    geometry_hash_by_system: Mapping[str, str],
    registered_operation_features_by_system: Mapping[str, np.ndarray],
    materialized_rows: Optional[Sequence[V8MeasuredRowV1]] = None,
    payload_scope: str = "all_trusted_rows",
    selection_before_payload_load: bool = False,
    raw_register_payloads_opened: bool = True,
) -> Dict[str, Any]:
    rows = tuple(bundle.rows if materialized_rows is None else materialized_rows)
    split_counts = Counter(row.split_name for row in rows)
    system_counts = Counter(row.system_type for row in rows)
    exposure_counts = Counter(
        f"{row.system_type}:{'known' if row.exposure_known_mask else 'unknown'}"
        for row in rows
    )
    rho_counts = Counter(
        f"{row.system_type}:{'available' if row.rho_available_mask else 'missing'}"
        for row in rows
    )
    locked_metadata_rows = tuple(
        row for row in bundle.rows if row.split_name not in DEVELOPMENT_SPLIT_NAMES
    )
    materialized_ids = {row.sample_id for row in rows}
    locked_payload_open_count = sum(
        row.sample_id in materialized_ids for row in locked_metadata_rows
    )
    feature_hashes = {
        system_type: hashlib.sha256(
            np.ascontiguousarray(features).view(np.uint8)
        ).hexdigest()
        for system_type, features in registered_operation_features_by_system.items()
    }
    return {
        "schema": "physics_hybrid_v8_dlcz_measured_data_audit_v1",
        "sample_count": len(rows),
        "sample_count_by_system": dict(sorted(system_counts.items())),
        "split_count": dict(sorted(split_counts.items())),
        "exposure_count": dict(sorted(exposure_counts.items())),
        "rho_count": dict(sorted(rho_counts.items())),
        "processed_count_shapes": {"3qubit": [75], "4qubit": [256]},
        "likelihood_count_source": "processed_counter_unwrapped_observed_not_noise_subtracted",
        "raw_register_role": "hash_bound_finite_width_counter_qa_only",
        "raw_registers_retained": bool(include_raw_registers),
        "payload_integrity_check_mode": str(integrity_check_mode),
        "processed_payload_sha256_verified": (
            integrity_check_mode == "full_payload_sha256"
        ),
        "manifest_entry_hashes_validated": True,
        "raw_register_counter_unwrap_qa_performed": bool(
            raw_register_payloads_opened
        ),
        "raw_register_payload_open_count": (
            len(rows) * len(RAW_COUNT_NAMES)
            if raw_register_payloads_opened
            else 0
        ),
        "processed_count_payload_open_count": len(rows) * len(PROCESSED_COUNT_NAMES),
        "rho_payload_open_count": (
            sum(row.rho_available_mask for row in rows)
            if include_rho_payloads
            else 0
        ),
        "rho_payload_loading_allowed": bool(include_rho_payloads),
        "snapshot_manifest_payload_open_count": len(rows),
        "payload_scope": str(payload_scope),
        "selection_before_payload_load": bool(selection_before_payload_load),
        "full_manifest_metadata_sample_count": len(bundle.rows),
        "materialized_sample_ids_sha256": _sample_ids_sha256(
            [row.sample_id for row in rows]
        ),
        "locked_final_metadata_row_count": len(locked_metadata_rows),
        "locked_final_payload_open_count": int(locked_payload_open_count),
        "locked_record_outcome_metadata_parsed": True,
        "strict_blind_final_claim": False,
        "counter_moduli": COUNTER_MODULI,
        "unwrapped_bin_count": dict(sorted(raw_unwrap_count.items())),
        "global_multi_scope": "global",
        "global_pair_labels": list(GLOBAL_PAIR_LABELS),
        "operation_local_multi_expansion_allowed": False,
        "real_noise_evidence_count": {name: 0 for name in REAL_NOISE_EVIDENCE_MASK_NAMES},
        "development_split_names": sorted(DEVELOPMENT_SPLIT_NAMES),
        "forbidden_development_split_tokens": list(FORBIDDEN_SPLIT_TOKENS),
        "registered_operation_feature_names": list(REGISTERED_OPERATION_FEATURE_NAMES),
        "registered_operation_feature_schema": REGISTERED_OPERATION_FEATURE_SCHEMA,
        "registered_operation_feature_provenance": REGISTERED_OPERATION_FEATURE_PROVENANCE,
        "registered_operation_feature_hash_by_system": feature_hashes,
        "geometry_hash_by_system": dict(sorted(geometry_hash_by_system.items())),
        "manifest_hash": bundle.manifest_hash,
        "split_hash": bundle.split_hash,
        "summary_sha256": bundle.summary_sha256,
        "operator_curated_corpus": dict(bundle.curated_corpus_audit),
        "stable_hash": stable_hash,
    }


def _group_components(
    data: V8DLCZMeasuredDataV1,
    indices: Sequence[int],
) -> List[Tuple[int, ...]]:
    selected = [int(index) for index in indices]
    parent = {index: index for index in selected}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    locations: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    for index in selected:
        system_type = data.system_types[index]
        locations[(system_type, "repeat", data.repeat_group_ids[index])].append(index)
        locations[(system_type, "time", data.time_block_ids[index])].append(index)
    for group in locations.values():
        first = group[0]
        for index in group[1:]:
            union(first, index)
    components: Dict[int, List[int]] = defaultdict(list)
    for index in selected:
        components[find(index)].append(index)
    return [tuple(sorted(component)) for component in components.values()]


def _validate_fold_group_isolation(
    data: V8DLCZMeasuredDataV1,
    selected: Sequence[int],
    folds: Sequence[V8DLCZCrossFitFoldV1],
) -> None:
    validation_location: Dict[int, int] = {}
    for fold in folds:
        train = set(int(index) for index in fold.train_indices)
        validation = set(int(index) for index in fold.validation_indices)
        if train & validation:
            raise ValueError("cross-fit train/validation overlap")
        for index in validation:
            if index in validation_location:
                raise ValueError("row appears in more than one validation fold")
            validation_location[index] = fold.fold_index
    if set(validation_location) != set(int(index) for index in selected):
        raise ValueError("cross-fit validation folds do not exactly cover selected rows")
    for label, values in (
        ("repeat", data.repeat_group_ids),
        ("time", data.time_block_ids),
    ):
        locations: Dict[Tuple[str, str], set[int]] = defaultdict(set)
        for index in selected:
            locations[(data.system_types[int(index)], values[int(index)])].add(
                validation_location[int(index)]
            )
        if any(len(fold_indices) != 1 for fold_indices in locations.values()):
            raise ValueError(f"{label} group crosses cross-fit validation folds")


def _load_hash_bound_integer_array(
    path: Path,
    expected_hash: str,
    *,
    expected_shape: Tuple[int, ...],
    field_name: str,
    verify_payload_hash: bool = True,
) -> np.ndarray:
    values = _load_hash_bound_array(
        path,
        expected_hash,
        field_name=field_name,
        verify_payload_hash=verify_payload_hash,
    )
    if values.shape != expected_shape:
        raise ValueError(f"{field_name} must have shape {expected_shape}, got {values.shape}")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError(f"{field_name} must contain real numeric counts")
    if values.dtype.kind in "iu":
        if values.dtype.kind == "i" and np.any(values < 0):
            raise ValueError(f"{field_name} contains negative counts")
        if values.dtype.kind == "u" and np.any(values > np.iinfo(np.int64).max):
            raise ValueError(f"{field_name} exceeds int64 count range")
        return np.asarray(values, dtype=np.int64)
    numeric = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(numeric)) or np.any(numeric < 0.0):
        raise ValueError(f"{field_name} contains negative or non-finite counts")
    if not np.all(numeric == np.floor(numeric)):
        raise ValueError(f"{field_name} contains non-integer count values")
    if np.any(numeric > np.iinfo(np.int64).max):
        raise ValueError(f"{field_name} exceeds int64 count range")
    return numeric.astype(np.int64)


def _load_hash_bound_array(
    path: Path,
    expected_hash: str,
    *,
    field_name: str,
    verify_payload_hash: bool = True,
) -> np.ndarray:
    _require_sha256(expected_hash, f"{field_name} hash")
    if not path.is_file():
        raise ValueError(f"array path is missing for {field_name}: {path}")
    if verify_payload_hash and _sha256_file(path) != expected_hash:
        raise ValueError(f"array path/hash mismatch for {field_name}: {path}")
    try:
        return np.asarray(np.load(path, allow_pickle=False))
    except Exception as exc:
        raise ValueError(f"cannot load hash-bound array for {field_name}: {path}") from exc


def _validate_counter_unwrap(
    processed: np.ndarray,
    raw: np.ndarray,
    *,
    modulus: int,
    field_name: str,
) -> None:
    if int(modulus) <= 1:
        raise ValueError("counter modulus must exceed one")
    if np.any(raw >= int(modulus)):
        raise ValueError(f"raw register exceeds its finite-width modulus for {field_name}")
    delta = processed - raw
    if np.any(delta < 0) or np.any(np.mod(delta, int(modulus)) != 0):
        raise ValueError(f"processed count is not a valid modulo unwrap for {field_name}")


def _validate_record_totals(record: Any, arrays: Mapping[str, np.ndarray]) -> None:
    expected = {
        "signal_bins": _nonnegative_integer_scalar(record.signal_total, "signal_total"),
        "idler_bins": _nonnegative_integer_scalar(record.idler_total, "idler_total"),
        "coin_bins": _nonnegative_integer_scalar(record.coin_total, "coin_total"),
    }
    for name, total in expected.items():
        if int(np.sum(arrays[name], dtype=np.int64)) != total:
            raise ValueError(f"manifest total disagrees with {name} for {record.sample_id}")


def _validate_density_matrix(
    density: np.ndarray,
    *,
    expected_dimension: int,
    field_name: str,
) -> np.ndarray:
    values = np.asarray(density, dtype=np.complex128)
    expected_shape = (int(expected_dimension), int(expected_dimension))
    if values.shape != expected_shape:
        raise ValueError(f"rho shape changed for {field_name}: expected {expected_shape}")
    if not np.all(np.isfinite(values.real)) or not np.all(np.isfinite(values.imag)):
        raise ValueError(f"rho contains non-finite values for {field_name}")
    if not np.allclose(values, values.conj().T, rtol=0.0, atol=1e-8):
        raise ValueError(f"rho is not Hermitian for {field_name}")
    trace = float(np.real(np.trace(values)))
    if abs(trace - 1.0) > 1e-6 or abs(float(np.imag(np.trace(values)))) > 1e-8:
        raise ValueError(f"rho trace is not one for {field_name}")
    if float(np.min(np.linalg.eigvalsh(values))) < -1e-8:
        raise ValueError(f"rho is not positive semidefinite for {field_name}")
    return _read_only(values)


def _nonnegative_integer_scalar(value: Any, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be an integer count")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer count") from exc
    if not np.isfinite(numeric) or numeric < 0.0 or numeric != np.floor(numeric):
        raise ValueError(f"{field_name} must be a non-negative integer")
    if numeric > np.iinfo(np.int64).max:
        raise ValueError(f"{field_name} exceeds int64 count range")
    return int(numeric)


def _validated_subset_indices(indices: Sequence[int], sample_count: int) -> np.ndarray:
    values = _validated_index_array(indices, "indices")
    if not values.size:
        raise ValueError("indices cannot be empty")
    if np.any(values >= int(sample_count)):
        raise IndexError("row index is outside the measured data")
    if len(set(int(value) for value in values)) != len(values):
        raise ValueError("row indices must be unique")
    return np.asarray(sorted(int(value) for value in values), dtype=np.int64)


def _validated_index_array(values: Sequence[int] | np.ndarray, field_name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"{field_name} must be a one-dimensional integer array")
    if np.any(array < 0):
        raise ValueError(f"{field_name} cannot contain negative indices")
    return array.astype(np.int64, copy=False)


def _sample_ids_sha256(sample_ids: Sequence[str]) -> str:
    return stable_json_sha256({"sample_ids": list(sample_ids)})


def _read_only(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    array.setflags(write=False)
    return array


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, field_name: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return text


__all__ = [
    "COUNTER_MODULI",
    "DEVELOPMENT_SPLIT_NAMES",
    "DLCZ_MEASURED_DATA_SCHEMA",
    "GLOBAL_PAIR_LABELS",
    "RAW_COUNT_NAMES",
    "REAL_NOISE_EVIDENCE_MASK_NAMES",
    "REGISTERED_OPERATION_FEATURE_NAMES",
    "REGISTERED_OPERATION_FEATURE_SCHEMA",
    "REGISTERED_OPERATION_FEATURE_PROVENANCE",
    "STATIC_PRIOR_PHYSICS_FEATURE_NAMES",
    "TRAIN_SPLIT_NAMES",
    "VALIDATION_SPLIT_NAMES",
    "V8DLCZCrossFitFoldV1",
    "V8DLCZDevelopmentIndicesV1",
    "V8DLCZMeasuredDataV1",
    "build_group_cross_fit_folds",
    "build_registered_operation_features",
    "build_v8_a0_development_contexts",
    "compact_dlcz_data_audit",
    "development_train_validation_indices",
    "load_v8_dlcz_development_data",
    "load_v8_dlcz_measured_data",
    "nb2_count_labels",
    "select_development_indices",
]
