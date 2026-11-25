"""Manifest-bound real-data preparation for V7 Model A and Model B.

Every trusted real row is retained.  Count-only rows remain useful for masked
operation modelling, while density losses are controlled by an explicit mask.
No missing density matrix is represented as an observed zero matrix.  The
legacy global multi-click counters are exposed only as interval upper bounds;
they are never expanded into operation-local pseudo-counts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ml_recon.measured_data.datasets import MeasuredManifestDataset, load_measured_records
from ml_recon.physics_identification.operation_point_calibration import (
    load_operation_point_dataset,
)
from system_control.data_ui.rho_fidelity_summary import w_state_indices

from .geometry import OperationFrequencyGeometryV1, build_operation_frequency_geometry
from .physical_ceiling import load_retrieval_rate_vectors


FEEDBACK_DATA_SCHEMA = "v7_feedback_measured_arrays_v1"
NUISANCE_PROFILE_SCHEMA = "v7_feedback_nuisance_profile_v1"
MAX_QUBITS = 4
MAX_ACTION_DIM = 6
COUNT_SMOOTHING = 0.5
MINIMUM_VARIANCE = 1e-6

OPERATION_FEATURE_NAMES = (
    "log_coin_observed",
    "log_signal",
    "log_idler",
    "gaussian_coherent_z",
    "gaussian_coherent_signed_log",
    "log_gsi_diagnostic",
    "log_read_count",
    "observed_coin_mask",
    "retrieval_relative_map",
    "target_born_overlap",
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
)

TARGET_FEATURE_NAMES = tuple(
    [f"target_amplitude_q{index + 1}" for index in range(MAX_QUBITS)]
    + [f"target_phase_sin_q{index + 1}" for index in range(MAX_QUBITS)]
    + [f"target_phase_cos_q{index + 1}" for index in range(MAX_QUBITS)]
    + [f"target_qubit_mask_q{index + 1}" for index in range(MAX_QUBITS)]
)


@dataclass(frozen=True)
class FeedbackMeasuredArraysV1:
    system_type: str
    sample_ids: Tuple[str, ...]
    split_names: np.ndarray
    repeat_group_ids: Tuple[str, ...]
    time_block_ids: Tuple[str, ...]
    coin_counts: np.ndarray
    signal_counts: np.ndarray
    idler_counts: np.ndarray
    read_count: np.ndarray
    target_amplitude: np.ndarray
    target_phase_rad: np.ndarray
    target_mask: np.ndarray
    rho_full: Tuple[Optional[np.ndarray], ...]
    rho_core: Tuple[Optional[np.ndarray], ...]
    rho_available_mask: np.ndarray
    sample_weight: np.ndarray
    multi_upper_rates: np.ndarray
    multi_interval_mask: np.ndarray
    static_operation_features: np.ndarray
    graph_kernel: np.ndarray
    retrieval_rate_vectors: np.ndarray
    retrieval_relative_map: np.ndarray
    geometry_hash: str
    manifest_hash: str

    @property
    def sample_count(self) -> int:
        return int(self.coin_counts.shape[0])

    @property
    def operation_count(self) -> int:
        return int(self.coin_counts.shape[1])

    @property
    def qubit_count(self) -> int:
        return 3 if self.system_type == "3qubit" else 4

    def split_mask(self, purpose: str) -> np.ndarray:
        names = self.split_names.astype(str)
        if purpose == "train":
            return np.asarray([
                name in {"train", "train_target_family", "probe_train"}
                for name in names
            ], dtype=bool)
        if purpose == "validation":
            return np.asarray([
                name in {"validation", "validation_target_family", "probe_validation"}
                for name in names
            ], dtype=bool)
        if purpose == "test_ood":
            return ~(self.split_mask("train") | self.split_mask("validation"))
        return names == str(purpose)


@dataclass(frozen=True)
class FeedbackNuisanceProfileV1:
    system_type: str
    operation_count: int
    train_sample_count: int
    train_sample_hash: str
    manifest_hash: str
    log_coin_total_mean: float
    log_coin_total_std: float
    log_signal_total_mean: float
    log_signal_total_std: float
    log_idler_total_mean: float
    log_idler_total_std: float
    log_read_count_mean: float
    log_read_count_std: float
    nb2_dispersion: float
    coherent_fraction_quantiles: Tuple[float, float, float]
    signal_probability_mean: np.ndarray
    idler_probability_mean: np.ndarray
    signal_log_share_modes: np.ndarray
    idler_log_share_modes: np.ndarray
    retrieval_rate_vectors: np.ndarray
    retrieval_relative_map: np.ndarray
    multi_upper_rate_quantiles: np.ndarray
    phase_map_error_bound_rad: float = float(np.deg2rad(10.0))
    r_phase_primary_bound_rad: float = float(np.deg2rad(10.0))
    r_phase_sensitivity_bound_rad: float = float(np.deg2rad(20.0))
    signal_efficiency_lower: float = 0.06
    signal_efficiency_upper: float = 0.10
    read_path_atom_output_to_bs_input: float = 0.23
    retrieval_relative_uncertainty_floor: float = 0.10
    research_only: bool = True
    promotion_ready: bool = False

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for name in (
            "signal_probability_mean",
            "idler_probability_mean",
            "signal_log_share_modes",
            "idler_log_share_modes",
            "retrieval_rate_vectors",
            "retrieval_relative_map",
            "multi_upper_rate_quantiles",
        ):
            payload[name] = np.asarray(payload[name]).tolist()
        payload["schema"] = NUISANCE_PROFILE_SCHEMA
        payload["profile_hash"] = self.profile_hash
        payload["legacy_multi_contract"] = {
            "scope": "global",
            "semantic": "interval_censored_upper_bound_only",
            "operation_local_expansion_allowed": False,
            "maximum_training_loss_fraction": 0.01,
        }
        return payload

    @property
    def profile_hash(self) -> str:
        payload = asdict(self)
        for name, value in list(payload.items()):
            if isinstance(value, np.ndarray):
                payload[name] = value.tolist()
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def load_feedback_measured_arrays(
    *,
    system_type: str,
    manifest_dir: str | Path = "ml_recon/reports/measured_data",
    data_root: str | Path = "data",
) -> FeedbackMeasuredArraysV1:
    """Load every trusted row and preserve the repository split assignment."""

    manifest_path = Path(manifest_dir)
    dataset = load_operation_point_dataset(
        manifest_path,
        data_root=data_root,
        system_type=system_type,
    )
    records = dataset.records
    strict_dataset = MeasuredManifestDataset(records, data_root=data_root)
    if tuple(strict_dataset.records) != tuple(records):
        raise ValueError("strict dataset ordering disagrees with operation-point dataset")
    geometry = build_operation_frequency_geometry(system_type)
    if dataset.coin_counts.shape[1] != geometry.operation_count:
        raise ValueError("measured counts and operation geometry disagree")
    retrieval_vectors, _ = load_retrieval_rate_vectors(system_type, geometry.operation_count)
    retrieval_vectors = np.asarray(retrieval_vectors, dtype=np.float64)
    expected_retrieval_shape = (geometry.operation_count, geometry.qubit_count)
    if retrieval_vectors.shape != expected_retrieval_shape:
        raise ValueError(
            f"retrieval vectors must have shape {expected_retrieval_shape}, "
            f"got {retrieval_vectors.shape}"
        )
    retrieval_vectors /= max(float(np.max(retrieval_vectors)), 1e-12)
    retrieval = np.mean(retrieval_vectors, axis=1)
    retrieval /= max(float(np.max(retrieval)), 1e-12)
    static_features, static_names = geometry.static_features()
    if tuple(static_names) != OPERATION_FEATURE_NAMES[10:]:
        raise ValueError("operation geometry feature schema changed")
    graph_kernel = geometry.coupling_kernel(
        x_length_mhz=8.0,
        y_length_mhz=8.0,
        same_local_cross_cycle_weight=0.05,
    )
    graph_kernel += np.eye(geometry.operation_count, dtype=np.float64)
    graph_kernel /= np.maximum(graph_kernel.sum(axis=1, keepdims=True), 1e-12)

    target_amplitude = np.zeros((len(records), MAX_QUBITS), dtype=np.float32)
    target_phase = np.zeros_like(target_amplitude)
    target_mask = np.zeros_like(target_amplitude)
    rho_full: List[Optional[np.ndarray]] = []
    rho_core: List[Optional[np.ndarray]] = []
    rho_available = np.zeros(len(records), dtype=np.float32)
    read_count = np.zeros(len(records), dtype=np.float64)
    sample_weight = np.zeros(len(records), dtype=np.float32)
    multi_upper = np.zeros((len(records), 3), dtype=np.float64)
    multi_mask = np.zeros(len(records), dtype=np.float32)
    qubit_count = geometry.qubit_count
    core_indices = w_state_indices(qubit_count)

    for index, record in enumerate(records):
        amplitude = np.asarray(record.exact_target_state_params.get("amplitudes", []), dtype=np.float64)
        phase = np.asarray(record.exact_target_state_params.get("phases_rad", []), dtype=np.float64)
        if amplitude.shape != (qubit_count,) or phase.shape != amplitude.shape:
            raise ValueError(f"target state is incomplete for {record.sample_id}")
        amplitude = np.abs(amplitude)
        amplitude /= max(float(np.linalg.norm(amplitude)), 1e-12)
        target_amplitude[index, :qubit_count] = amplitude.astype(np.float32)
        target_phase[index, :qubit_count] = np.angle(np.exp(1j * phase)).astype(np.float32)
        target_mask[index, :qubit_count] = 1.0
        read_count[index] = max(float(record.read_count), 1.0)
        sample_weight[index] = float(record.sample_weight)
        multi_upper[index] = np.asarray(
            [record.multi2_count, record.multi3_count, record.multi4_count],
            dtype=np.float64,
        ) / read_count[index]
        if record.multi_click_scope == "global":
            multi_mask[index] = 1.0
        if record.rho_available:
            row = strict_dataset[index]
            density = np.asarray(row["rho"], dtype=np.complex128)
            density = 0.5 * (density + density.conj().T)
            full_trace = float(np.real(np.trace(density)))
            if full_trace <= 1e-10:
                raise ValueError(f"full rho has zero trace for {record.sample_id}")
            density = density / full_trace
            core = density[np.ix_(core_indices, core_indices)]
            core = 0.5 * (core + core.conj().T)
            trace = float(np.real(np.trace(core)))
            if trace <= 1e-10:
                raise ValueError(f"rho core has zero trace for {record.sample_id}")
            core = core / trace
            rho_full.append(density.astype(np.complex64))
            rho_core.append(core.astype(np.complex64))
            rho_available[index] = float(record.rho_branch_mask)
        else:
            rho_full.append(None)
            rho_core.append(None)

    manifest_summary = json.loads(
        (manifest_path / "measured_data_summary.json").read_text(encoding="utf-8")
    )
    manifest_hash = str(manifest_summary.get("manifest_hash") or "")
    if len(manifest_hash) != 64:
        raise ValueError("measured-data manifest hash is missing")
    return FeedbackMeasuredArraysV1(
        system_type=system_type,
        sample_ids=tuple(record.sample_id for record in records),
        split_names=np.asarray(dataset.split_names, dtype=str),
        repeat_group_ids=tuple(record.repeat_group_id for record in records),
        time_block_ids=tuple(record.time_block_id for record in records),
        coin_counts=np.asarray(dataset.coin_counts, dtype=np.float32),
        signal_counts=np.asarray(dataset.signal_counts, dtype=np.float32),
        idler_counts=np.asarray(dataset.idler_counts, dtype=np.float32),
        read_count=read_count.astype(np.float32),
        target_amplitude=target_amplitude,
        target_phase_rad=target_phase,
        target_mask=target_mask,
        rho_full=tuple(rho_full),
        rho_core=tuple(rho_core),
        rho_available_mask=rho_available,
        sample_weight=sample_weight,
        multi_upper_rates=multi_upper.astype(np.float32),
        multi_interval_mask=multi_mask,
        static_operation_features=static_features.astype(np.float32),
        graph_kernel=graph_kernel.astype(np.float32),
        retrieval_rate_vectors=retrieval_vectors.astype(np.float32),
        retrieval_relative_map=retrieval.astype(np.float32),
        geometry_hash=geometry.geometry_hash,
        manifest_hash=manifest_hash,
    )


def fit_feedback_nuisance_profile(
    arrays: FeedbackMeasuredArraysV1,
    *,
    maximum_modes: int = 8,
) -> FeedbackNuisanceProfileV1:
    """Fit synthetic-generator nuisance statistics from train groups only."""

    train_mask = arrays.split_mask("train")
    if not np.any(train_mask):
        raise ValueError("no train rows are available")
    coin = arrays.coin_counts[train_mask].astype(np.float64)
    signal = arrays.signal_counts[train_mask].astype(np.float64)
    idler = arrays.idler_counts[train_mask].astype(np.float64)
    read = arrays.read_count[train_mask].astype(np.float64)
    signal_probability = _smoothed_probability(signal)
    idler_probability = _smoothed_probability(idler)
    signal_modes = _log_share_modes(signal_probability, int(maximum_modes))
    idler_modes = _log_share_modes(idler_probability, int(maximum_modes))
    accidental = signal * idler / np.maximum(read[:, None], 1.0)
    coherent = coin - accidental
    coherent_fraction = np.sum(np.clip(coherent, 0.0, None), axis=1) / np.maximum(
        np.sum(coin, axis=1), 1.0,
    )
    train_indices = np.flatnonzero(train_mask)
    dispersion = _repeat_group_nb2_dispersion(arrays, train_indices)
    train_hash = hashlib.sha256(
        "\n".join(arrays.sample_ids[index] for index in train_indices).encode("utf-8")
    ).hexdigest()
    multi_quantiles = np.quantile(
        arrays.multi_upper_rates[train_mask],
        [0.50, 0.90, 0.99],
        axis=0,
    )
    return FeedbackNuisanceProfileV1(
        system_type=arrays.system_type,
        operation_count=arrays.operation_count,
        train_sample_count=int(np.sum(train_mask)),
        train_sample_hash=train_hash,
        manifest_hash=arrays.manifest_hash,
        log_coin_total_mean=float(np.mean(np.log1p(np.sum(coin, axis=1)))),
        log_coin_total_std=float(max(np.std(np.log1p(np.sum(coin, axis=1))), 0.05)),
        log_signal_total_mean=float(np.mean(np.log1p(np.sum(signal, axis=1)))),
        log_signal_total_std=float(max(np.std(np.log1p(np.sum(signal, axis=1))), 0.05)),
        log_idler_total_mean=float(np.mean(np.log1p(np.sum(idler, axis=1)))),
        log_idler_total_std=float(max(np.std(np.log1p(np.sum(idler, axis=1))), 0.05)),
        log_read_count_mean=float(np.mean(np.log1p(read))),
        log_read_count_std=float(max(np.std(np.log1p(read)), 0.02)),
        nb2_dispersion=float(dispersion),
        coherent_fraction_quantiles=tuple(
            float(value) for value in np.quantile(coherent_fraction, [0.10, 0.50, 0.90])
        ),
        signal_probability_mean=np.mean(signal_probability, axis=0).astype(np.float32),
        idler_probability_mean=np.mean(idler_probability, axis=0).astype(np.float32),
        signal_log_share_modes=signal_modes.astype(np.float32),
        idler_log_share_modes=idler_modes.astype(np.float32),
        retrieval_rate_vectors=arrays.retrieval_rate_vectors.astype(np.float32),
        retrieval_relative_map=arrays.retrieval_relative_map.astype(np.float32),
        multi_upper_rate_quantiles=multi_quantiles.astype(np.float32),
    )


def build_target_features(
    amplitude: np.ndarray,
    phase_rad: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    return np.concatenate((
        np.asarray(amplitude, dtype=np.float32),
        np.sin(np.asarray(phase_rad, dtype=np.float32)),
        np.cos(np.asarray(phase_rad, dtype=np.float32)),
        np.asarray(target_mask, dtype=np.float32),
    ), axis=-1)


def gaussian_coherent_residual(
    coin: np.ndarray,
    signal: np.ndarray,
    idler: np.ndarray,
    read_count: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``Y=C-SI/N`` and its multinomial delta-method variance."""

    c = np.asarray(coin, dtype=np.float64)
    s = np.asarray(signal, dtype=np.float64)
    i = np.asarray(idler, dtype=np.float64)
    n = np.asarray(read_count, dtype=np.float64)
    while n.ndim < c.ndim:
        n = n[..., None]
    n = np.maximum(n, 1.0)
    ps = np.clip(s / n, 0.0, 1.0)
    pi = np.clip(i / n, 0.0, 1.0)
    pc = np.clip(c / n, 0.0, np.minimum(ps, pi))
    variance_s = n * ps * (1.0 - ps)
    variance_i = n * pi * (1.0 - pi)
    variance_c = n * pc * (1.0 - pc)
    covariance_si = n * (pc - ps * pi)
    covariance_sc = n * pc * (1.0 - ps)
    covariance_ic = n * pc * (1.0 - pi)
    gradient_s = -i / n
    gradient_i = -s / n
    variance = (
        np.square(gradient_s) * variance_s
        + np.square(gradient_i) * variance_i
        + variance_c
        + 2.0 * gradient_s * gradient_i * covariance_si
        + 2.0 * gradient_s * covariance_sc
        + 2.0 * gradient_i * covariance_ic
    )
    residual = c - s * i / n
    return residual.astype(np.float32), np.maximum(variance, MINIMUM_VARIANCE).astype(np.float32)


def build_operation_features(
    *,
    coin_counts: np.ndarray,
    signal_counts: np.ndarray,
    idler_counts: np.ndarray,
    read_count: np.ndarray,
    target_amplitude: np.ndarray,
    target_phase_rad: np.ndarray,
    geometry: OperationFrequencyGeometryV1,
    static_operation_features: np.ndarray,
    retrieval_relative_map: np.ndarray,
    observed_coin_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    coin = np.asarray(coin_counts, dtype=np.float32)
    signal = np.asarray(signal_counts, dtype=np.float32)
    idler = np.asarray(idler_counts, dtype=np.float32)
    if coin.ndim != 2 or signal.shape != coin.shape or idler.shape != coin.shape:
        raise ValueError("count arrays must have matching [batch, operation] shapes")
    batch, operation_count = coin.shape
    if operation_count != geometry.operation_count:
        raise ValueError("counts and geometry disagree")
    visible = np.ones_like(coin, dtype=np.float32) if observed_coin_mask is None else np.asarray(
        observed_coin_mask, dtype=np.float32,
    )
    if visible.shape != coin.shape:
        raise ValueError("observed_coin_mask shape differs from counts")
    residual, variance = gaussian_coherent_residual(coin, signal, idler, read_count)
    signed_log = np.sign(residual) * np.log1p(np.abs(residual))
    coherent_z = residual / np.sqrt(variance)
    read = np.asarray(read_count, dtype=np.float32).reshape(batch, 1)
    log_gsi = (
        np.log(coin + COUNT_SMOOTHING)
        + np.log(np.maximum(read, 1.0))
        - np.log(signal + COUNT_SMOOTHING)
        - np.log(idler + COUNT_SMOOTHING)
    )
    overlap = np.stack([
        geometry.target_measurement_overlap({
            "amplitudes": target_amplitude[index, :geometry.qubit_count],
            "phases_rad": target_phase_rad[index, :geometry.qubit_count],
        })
        for index in range(batch)
    ]).astype(np.float32)
    dynamic = np.stack((
        np.log1p(coin) * visible,
        np.log1p(signal),
        np.log1p(idler),
        coherent_z * visible,
        signed_log * visible,
        log_gsi * visible,
        np.broadcast_to(np.log1p(read), coin.shape),
        visible,
        np.broadcast_to(np.asarray(retrieval_relative_map, dtype=np.float32), coin.shape),
        overlap,
    ), axis=-1)
    static = np.broadcast_to(
        np.asarray(static_operation_features, dtype=np.float32)[None, :, :],
        (batch, operation_count, static_operation_features.shape[1]),
    )
    features = np.concatenate((dynamic, static), axis=-1)
    if features.shape[-1] != len(OPERATION_FEATURE_NAMES):
        raise ValueError("operation feature schema changed")
    if not np.all(np.isfinite(features)):
        raise ValueError("operation features contain non-finite values")
    return features.astype(np.float32)


def collate_rho_core(
    arrays: FeedbackMeasuredArraysV1,
    indices: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Collate optional rho labels with an explicit mask and neutral sentinel.

    The identity sentinel exists only to keep tensor shapes finite.  It is
    always paired with mask zero and must never be counted as observed data.
    """

    qubit_count = arrays.qubit_count
    values = np.empty((len(indices), qubit_count, qubit_count), dtype=np.complex64)
    mask = np.zeros(len(indices), dtype=np.float32)
    sentinel = np.eye(qubit_count, dtype=np.complex64) / float(qubit_count)
    for row, index in enumerate(indices):
        density = arrays.rho_core[int(index)]
        if density is None:
            values[row] = sentinel
        else:
            values[row] = density
            mask[row] = float(arrays.rho_available_mask[int(index)])
    return values, mask


def collate_rho_full(
    arrays: FeedbackMeasuredArraysV1,
    indices: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Collate full tomography rho with an explicit missing-data mask."""

    dimension = 2 ** arrays.qubit_count
    values = np.empty((len(indices), dimension, dimension), dtype=np.complex64)
    mask = np.zeros(len(indices), dtype=np.float32)
    sentinel = np.eye(dimension, dtype=np.complex64) / float(dimension)
    for row, index in enumerate(indices):
        density = arrays.rho_full[int(index)]
        if density is None:
            values[row] = sentinel
        else:
            values[row] = density
            mask[row] = float(arrays.rho_available_mask[int(index)])
    return values, mask


def deterministic_operation_mask(
    sample_ids: Sequence[str],
    operation_count: int,
    *,
    hidden_fraction: float,
    seed: int,
) -> np.ndarray:
    if not 0.0 < float(hidden_fraction) < 1.0:
        raise ValueError("hidden_fraction must lie strictly between zero and one")
    hidden_count = min(
        operation_count - 1,
        max(1, int(round(operation_count * float(hidden_fraction)))),
    )
    result = np.zeros((len(sample_ids), operation_count), dtype=np.float32)
    for row, sample_id in enumerate(sample_ids):
        digest = hashlib.sha256(f"{seed}|{sample_id}".encode("utf-8")).digest()
        local_seed = int.from_bytes(digest[:8], "little", signed=False)
        selected = np.random.default_rng(local_seed).choice(
            operation_count,
            size=hidden_count,
            replace=False,
        )
        result[row, selected] = 1.0
    return result


def _smoothed_probability(counts: np.ndarray) -> np.ndarray:
    value = np.asarray(counts, dtype=np.float64) + COUNT_SMOOTHING
    return value / np.maximum(value.sum(axis=1, keepdims=True), 1e-12)


def _log_share_modes(probability: np.ndarray, maximum_modes: int) -> np.ndarray:
    log_share = np.log(np.clip(probability, 1e-12, None))
    log_share -= np.mean(log_share, axis=0, keepdims=True)
    _, _, right = np.linalg.svd(log_share, full_matrices=False)
    mode_count = min(int(maximum_modes), right.shape[0])
    return right[:mode_count]


def _repeat_group_nb2_dispersion(
    arrays: FeedbackMeasuredArraysV1,
    train_indices: np.ndarray,
) -> float:
    groups: Dict[str, List[int]] = {}
    for index in train_indices:
        groups.setdefault(arrays.repeat_group_ids[int(index)], []).append(int(index))
    estimates: List[np.ndarray] = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        counts = arrays.coin_counts[indices].astype(np.float64)
        totals = np.maximum(counts.sum(axis=1), 1.0)
        reference_total = float(np.exp(np.mean(np.log(totals))))
        normalized = counts * (reference_total / totals)[:, None]
        mean = np.mean(normalized, axis=0)
        variance = np.var(normalized, axis=0, ddof=1)
        valid = mean >= 2.0
        if np.any(valid):
            estimates.append(np.clip((variance[valid] - mean[valid]) / np.square(mean[valid]), 0.0, 1.0))
    if not estimates:
        return 0.02
    combined = np.concatenate(estimates)
    return float(np.clip(np.median(combined), 1e-4, 0.5))


__all__ = [
    "COUNT_SMOOTHING",
    "FEEDBACK_DATA_SCHEMA",
    "FeedbackMeasuredArraysV1",
    "FeedbackNuisanceProfileV1",
    "MAX_ACTION_DIM",
    "MAX_QUBITS",
    "NUISANCE_PROFILE_SCHEMA",
    "OPERATION_FEATURE_NAMES",
    "TARGET_FEATURE_NAMES",
    "build_operation_features",
    "build_target_features",
    "collate_rho_core",
    "deterministic_operation_mask",
    "fit_feedback_nuisance_profile",
    "gaussian_coherent_residual",
    "load_feedback_measured_arrays",
]
