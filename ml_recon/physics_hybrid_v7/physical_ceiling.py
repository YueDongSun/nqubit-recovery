"""Physics-factor encoding and conformal best-fidelity ceilings for V7.

The module is research-only.  It consumes immutable measured snapshots and
versioned retrieval calibration, but it cannot materialize commands or write
hardware deltas.  Two ceilings are kept deliberately separate:

* ``physics_hypothesis_ceiling`` follows the relaxed GSI mixture hypothesis and
  retrieval imbalance model;
* ``statistical_upper_bound`` is a one-sided split-conformal bound learned from
  measured best-fidelity groups.

Disagreement between them is evidence against the physical assumptions, not a
reason to overwrite the measured result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from system_control.qubit_recon_files.system_specs import get_spec

from .geometry import OperationFrequencyGeometryV1


PHYSICS_CEILING_SCHEMA = "v7_physics_ceiling_v1"
BASE_PHYSICS_FACTOR_NAMES = (
    "gsi_proxy_bounded",
    "gsi_coherent_fraction",
    "gsi_white_mixture_ceiling",
    "retrieval_target_ceiling",
    "retrieval_target_worst_operation",
    "retrieval_component_cv",
    "combined_gsi_retrieval_ceiling",
    "signal_response_residual_rms",
    "idler_response_residual_rms",
    "coin_response_residual_rms",
    "signal_graph_roughness",
    "idler_graph_roughness",
    "coin_graph_roughness",
    "signal_idler_residual_correlation",
    "log_coin_total",
    "log_singles_geomean",
    "absolute_log_signal_idler_ratio",
    "global_multi_click_rate",
    "inverse_sqrt_coin_total",
    "log_raw_gsi_proxy",
    "gsi_clipping_excess",
)
RESPONSE_MODE_COUNT = 6
RESPONSE_MODE_NAMES = tuple(
    f"{channel}_response_mode_{mode_index + 1}"
    for channel in ("signal", "idler", "coin")
    for mode_index in range(RESPONSE_MODE_COUNT)
)
PHYSICS_FACTOR_NAMES = BASE_PHYSICS_FACTOR_NAMES + RESPONSE_MODE_NAMES


def _soft_normalized_log_share(counts: Any, smoothing: float) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    if values.ndim != 2 or np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("counts must be a finite non-negative two-dimensional array")
    share = (values + float(smoothing)) / (
        values.sum(axis=1, keepdims=True) + float(smoothing) * values.shape[1]
    )
    logged = np.log(np.maximum(share, 1e-12))
    return logged - logged.mean(axis=1, keepdims=True)


def _ridge_solution(
    features: np.ndarray,
    targets: np.ndarray,
    ridge_alpha: float,
    *,
    coefficient_prior: Optional[np.ndarray] = None,
    prior_weight: float = 0.0,
) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    penalty = float(ridge_alpha) * np.eye(x.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    rhs = x.T @ y
    if coefficient_prior is not None and float(prior_weight) > 0.0:
        prior = np.asarray(coefficient_prior, dtype=float)
        if prior.shape != (x.shape[1],):
            raise ValueError("coefficient_prior has the wrong shape")
        prior_penalty = float(prior_weight) * np.eye(x.shape[1], dtype=float)
        prior_penalty[0, 0] = 0.0
        penalty += prior_penalty
        rhs += prior_penalty @ prior
    return np.linalg.pinv(x.T @ x + penalty) @ rhs


def _equal_group_weights(group_ids: Any, sample_mask: Any) -> np.ndarray:
    groups = np.asarray(group_ids).astype(str)
    mask = np.asarray(sample_mask, dtype=bool)
    weights = np.zeros(groups.size, dtype=float)
    for group_id in np.unique(groups[mask]):
        selected = mask & (groups == group_id)
        weights[selected] = 1.0 / max(int(np.sum(selected)), 1)
    return weights


def _group_max_values(
    prediction: Any,
    labels: Any,
    group_ids: Any,
    sample_mask: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = np.asarray(prediction, dtype=float).reshape(-1)
    observed = np.asarray(labels, dtype=float).reshape(-1)
    groups = np.asarray(group_ids).astype(str)
    mask = np.asarray(sample_mask, dtype=bool)
    unique_groups = np.unique(groups[mask])
    predicted_best = np.asarray([
        np.max(predicted[mask & (groups == group_id)]) for group_id in unique_groups
    ])
    observed_best = np.asarray([
        np.max(observed[mask & (groups == group_id)]) for group_id in unique_groups
    ])
    return predicted_best, observed_best, unique_groups


def pure_target_fidelity(
    rho: Any,
    target: Mapping[str, Any],
    system_type: str,
) -> float:
    """Return ``<psi_target|rho|psi_target>`` in repository basis order."""

    num_qubits = 3 if system_type == "3qubit" else 4
    matrix = np.asarray(rho, dtype=complex)
    if matrix.shape != (2 ** num_qubits, 2 ** num_qubits):
        raise ValueError("rho shape does not match system_type")
    amplitudes = np.asarray(target.get("amplitudes", []), dtype=float).reshape(-1)
    phases = np.asarray(target.get("phases_rad", []), dtype=float).reshape(-1)
    if amplitudes.shape != (num_qubits,) or phases.shape != amplitudes.shape:
        raise ValueError("target state does not match system_type")
    amplitudes /= max(float(np.linalg.norm(amplitudes)), 1e-12)
    state = np.zeros(2 ** num_qubits, dtype=complex)
    indices = [2 ** (num_qubits - 1 - index) for index in range(num_qubits)]
    state[indices] = amplitudes * np.exp(1j * phases)
    return float(np.clip(np.real(np.vdot(state, matrix @ state)), 0.0, 1.0))


def load_retrieval_rate_vectors(
    system_type: str,
    operation_count: int,
    efficiency_map_path: Optional[str | Path] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load versioned retrieval efficiencies and map them to operations."""

    spec = get_spec(system_type)
    source = Path(efficiency_map_path) if efficiency_map_path is not None else Path(
        "system_control/qubit_recon_files/matrix"
    ) / f"retrieval_efficiency_{3 if system_type == '3qubit' else 4}qubit.txt"
    if not source.is_file():
        raise FileNotFoundError(f"versioned retrieval-efficiency map missing: {source}")
    efficiency_map = np.loadtxt(source, dtype=float)
    expected_shape = tuple(int(value) for value in spec["efficiency_shape"])
    if efficiency_map.shape != expected_shape:
        raise ValueError(
            f"{system_type} retrieval map must have shape {expected_shape}, got {efficiency_map.shape}"
        )
    if np.any(~np.isfinite(efficiency_map)) or np.any(efficiency_map <= 0.0):
        raise ValueError("retrieval ceiling requires finite strictly positive calibration rates")
    group_size = int(spec["group_size"])
    groups_per_row = int(spec["groups_per_row"])
    entries_per_cycle = int(spec["entries_per_cycle"])
    vectors = []
    for operation_index in range(int(operation_count)):
        local_index = operation_index % entries_per_cycle
        row = local_index // groups_per_row
        group = local_index % groups_per_row
        column_start = group * group_size
        vectors.append(efficiency_map[row, column_start:column_start + group_size])
    vectors = np.asarray(vectors, dtype=float)
    if np.any(vectors <= 0.0):
        raise ValueError("retrieval ceiling requires strictly positive calibration rates")
    return vectors.astype(np.float64), {
        "source_path": str(source),
        "system_type": system_type,
        "map_shape": list(efficiency_map.shape),
        "map_min": float(np.min(efficiency_map)),
        "map_max": float(np.max(efficiency_map)),
        "map_mean": float(np.mean(efficiency_map)),
        "field_amplitude_semantics": "sqrt_count_efficiency",
        "historical_snapshot_hash_available": False,
    }


def retrieval_ceiling_features(
    target: Mapping[str, Any],
    retrieval_rate_vectors: Any,
    geometry: OperationFrequencyGeometryV1,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Quantify target distortion caused by relative retrieval imbalance.

    For target populations ``p_i`` and operation-local efficiencies ``eta_i``,
    the coherent state overlap after amplitude attenuation is
    ``(sum p_i sqrt(eta_i))^2 / sum p_i eta_i``.  Absolute loss is intentionally
    excluded because the present measured corpus does not identify an absolute
    preparation rate from the conditional tomography pattern.
    """

    rates = np.asarray(retrieval_rate_vectors, dtype=float)
    if rates.shape != (geometry.operation_count, geometry.qubit_count):
        raise ValueError("retrieval_rate_vectors disagree with geometry")
    amplitudes = np.asarray(target.get("amplitudes", []), dtype=float).reshape(-1)
    if amplitudes.shape != (geometry.qubit_count,):
        raise ValueError("target amplitudes disagree with geometry")
    populations = np.square(amplitudes)
    populations /= max(float(np.sum(populations)), 1e-12)
    normalized_rates = rates / max(float(np.max(rates)), 1e-12)
    numerator = np.square(np.sum(populations[None, :] * np.sqrt(normalized_rates), axis=1))
    denominator = np.sum(populations[None, :] * normalized_rates, axis=1)
    operation_fidelity = np.clip(numerator / np.maximum(denominator, 1e-12), 0.0, 1.0)
    overlap_weight = np.asarray(geometry.target_measurement_overlap(target), dtype=float)
    overlap_weight = overlap_weight + 1e-6
    overlap_weight /= np.sum(overlap_weight)
    weighted_ceiling = float(np.sum(overlap_weight * operation_fidelity))
    weighted_mean = np.sum(populations[None, :] * normalized_rates, axis=1)
    weighted_variance = np.sum(
        populations[None, :] * np.square(normalized_rates - weighted_mean[:, None]), axis=1,
    )
    component_cv = np.sqrt(np.maximum(weighted_variance, 0.0)) / np.maximum(weighted_mean, 1e-12)
    values = np.asarray((
        weighted_ceiling,
        float(np.quantile(operation_fidelity, 0.10)),
        float(np.sum(overlap_weight * component_cv)),
    ), dtype=np.float64)
    return values, {
        "retrieval_target_ceiling": weighted_ceiling,
        "retrieval_target_worst_operation": float(np.min(operation_fidelity)),
        "retrieval_target_q10_operation": float(np.quantile(operation_fidelity, 0.10)),
        "retrieval_component_cv": float(np.sum(overlap_weight * component_cv)),
    }


@dataclass(frozen=True)
class PhysicsResponseSurfaceV1:
    """Training-only smooth AOD/retrieval response and GSI reference."""

    system_type: str
    geometry_hash: str
    operation_design_names: Tuple[str, ...]
    expected_log_share: np.ndarray
    operation_contrast_center: np.ndarray
    graph_smoother: np.ndarray
    response_modes: np.ndarray
    retrieval_rate_vectors: np.ndarray
    count_smoothing: float
    gsi_lower_bound: float
    gsi_upper_bound: float
    ideal_gsi: float
    training_sample_count: int
    research_only: bool = True
    promotion_ready: bool = False
    hardware_action_authorized: bool = False

    def transform(
        self,
        coin_counts: Any,
        signal_counts: Any,
        idler_counts: Any,
        targets: Sequence[Mapping[str, Any]],
        geometry: OperationFrequencyGeometryV1,
        *,
        multi_click_counts: Optional[Any] = None,
        gsi_coin_counts: Optional[Any] = None,
    ) -> Tuple[np.ndarray, Tuple[str, ...], Dict[str, np.ndarray]]:
        if geometry.geometry_hash != self.geometry_hash:
            raise ValueError("response surface and geometry hash disagree")
        coin = np.asarray(coin_counts, dtype=float)
        signal = np.asarray(signal_counts, dtype=float)
        idler = np.asarray(idler_counts, dtype=float)
        if coin.ndim != 2 or coin.shape != signal.shape or coin.shape != idler.shape:
            raise ValueError("coin/signal/idler counts must have the same shape")
        if coin.shape[1] != geometry.operation_count or len(targets) != coin.shape[0]:
            raise ValueError("counts/targets disagree with operation geometry")
        logged = np.stack([
            _soft_normalized_log_share(signal, self.count_smoothing),
            _soft_normalized_log_share(idler, self.count_smoothing),
            _soft_normalized_log_share(coin, self.count_smoothing),
        ], axis=1)
        residual = logged - self.expected_log_share[None, :, :]
        residual -= residual.mean(axis=2, keepdims=True)
        residual_rms = np.sqrt(np.mean(np.square(residual), axis=2))
        high_frequency = residual - np.einsum("oj,ncj->nco", self.graph_smoother, residual)
        roughness = np.sqrt(np.mean(np.square(high_frequency), axis=2))
        mode_scores = np.einsum("nco,ro->ncr", residual, self.response_modes)
        mode_scores /= np.sqrt(float(geometry.operation_count))
        residual_correlation = np.asarray([
            np.corrcoef(row[0], row[1])[0, 1]
            if np.std(row[0]) > 1e-12 and np.std(row[1]) > 1e-12 else 0.0
            for row in residual
        ])

        gsi_coin = coin if gsi_coin_counts is None else np.asarray(gsi_coin_counts, dtype=float)
        if gsi_coin.shape != coin.shape or np.any(gsi_coin < 0.0):
            raise ValueError("gsi_coin_counts must be non-negative and match coin_counts")
        coin_share = (gsi_coin + self.count_smoothing) / (
            gsi_coin.sum(axis=1, keepdims=True) + self.count_smoothing * gsi_coin.shape[1]
        )
        accidental = (signal + self.count_smoothing) * (idler + self.count_smoothing)
        accidental /= np.maximum(accidental.sum(axis=1, keepdims=True), 1e-12)
        lift = coin_share / np.maximum(accidental, 1e-12)
        normalized_lift = lift / np.maximum(self.operation_contrast_center, 1e-12)
        tail_low = np.quantile(normalized_lift, 0.10, axis=1)
        tail_high = np.quantile(normalized_lift, 0.90, axis=1)
        center_low = np.quantile(normalized_lift, 0.25, axis=1)
        center_high = np.quantile(normalized_lift, 0.75, axis=1)
        raw_gsi = np.sqrt(
            tail_high / np.maximum(tail_low, 1e-12)
            * center_high / np.maximum(center_low, 1e-12)
        )
        bounded_gsi = np.clip(raw_gsi, self.gsi_lower_bound, self.gsi_upper_bound)
        coherent_fraction = np.clip(
            (bounded_gsi - 1.0) / max(self.ideal_gsi - 1.0, 1e-12), 0.0, 1.0,
        )
        num_qubits = geometry.qubit_count
        gsi_ceiling = coherent_fraction + (1.0 - coherent_fraction) / num_qubits

        retrieval_rows = [
            retrieval_ceiling_features(target, self.retrieval_rate_vectors, geometry)
            for target in targets
        ]
        retrieval_values = np.stack([row[0] for row in retrieval_rows])
        retrieval_ceiling = retrieval_values[:, 0]
        combined_ceiling = (
            coherent_fraction * retrieval_ceiling
            + (1.0 - coherent_fraction) / num_qubits
        )
        signal_total = np.sum(signal, axis=1)
        idler_total = np.sum(idler, axis=1)
        coin_total = np.sum(coin, axis=1)
        singles_geomean = np.sqrt(np.maximum(signal_total * idler_total, 1.0))
        if multi_click_counts is None:
            multi_rate = np.zeros(coin.shape[0], dtype=float)
        else:
            multi = np.asarray(multi_click_counts, dtype=float)
            if multi.ndim != 2 or multi.shape[0] != coin.shape[0]:
                raise ValueError("multi_click_counts must have one row per sample")
            multi_rate = np.sum(np.maximum(multi, 0.0), axis=1) / (singles_geomean + 1.0)
        factors = np.column_stack((
            bounded_gsi,
            coherent_fraction,
            gsi_ceiling,
            retrieval_values[:, 0],
            retrieval_values[:, 1],
            retrieval_values[:, 2],
            combined_ceiling,
            residual_rms[:, 0],
            residual_rms[:, 1],
            residual_rms[:, 2],
            roughness[:, 0],
            roughness[:, 1],
            roughness[:, 2],
            residual_correlation,
            np.log1p(coin_total),
            np.log1p(singles_geomean),
            np.abs(np.log((signal_total + 1.0) / (idler_total + 1.0))),
            multi_rate,
            1.0 / np.sqrt(coin_total + 1.0),
            np.log(np.maximum(raw_gsi, 1e-12)),
            np.maximum(raw_gsi - bounded_gsi, 0.0),
            mode_scores.reshape(coin.shape[0], -1),
        ))
        diagnostics = {
            "raw_gsi_proxy": raw_gsi.astype(np.float64),
            "bounded_gsi_proxy": bounded_gsi.astype(np.float64),
            "physics_hypothesis_ceiling": combined_ceiling.astype(np.float64),
            "retrieval_ceiling": retrieval_ceiling.astype(np.float64),
            "response_residual_rms": residual_rms.astype(np.float64),
            "response_graph_roughness": roughness.astype(np.float64),
            "response_mode_scores": mode_scores.astype(np.float64),
        }
        return factors.astype(np.float64), PHYSICS_FACTOR_NAMES, diagnostics

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for name in (
            "expected_log_share", "operation_contrast_center",
            "graph_smoother", "response_modes", "retrieval_rate_vectors",
        ):
            payload[name] = np.asarray(payload[name]).tolist()
        payload["schema"] = "v7_physics_response_surface_v1"
        payload["inference_contract"] = (
            "training-only smooth response and contrast references; measured test coin counts "
            "are allowed only for retrospective ceiling analysis, never hidden-point feedback"
        )
        return payload


def fit_physics_response_surface(
    system_type: str,
    coin_counts: Any,
    signal_counts: Any,
    idler_counts: Any,
    train_mask: Any,
    geometry: OperationFrequencyGeometryV1,
    retrieval_rate_vectors: Any,
    *,
    count_smoothing: float = 0.5,
    response_ridge: float = 1.0,
    graph_length_mhz: float = 8.0,
    response_mode_count: int = RESPONSE_MODE_COUNT,
    gsi_lower_bound: float = 1.2,
    gsi_upper_bound: float = 1.5,
    ideal_gsi: float = 2.0,
    gsi_coin_counts: Optional[Any] = None,
) -> Tuple[PhysicsResponseSurfaceV1, Dict[str, Any]]:
    """Fit a positive, smooth operation response from training rows only."""

    coin = np.asarray(coin_counts, dtype=float)
    signal = np.asarray(signal_counts, dtype=float)
    idler = np.asarray(idler_counts, dtype=float)
    train = np.asarray(train_mask, dtype=bool).reshape(-1)
    if coin.ndim != 2 or coin.shape != signal.shape or coin.shape != idler.shape:
        raise ValueError("coin/signal/idler counts must have the same shape")
    if train.shape != (coin.shape[0],) or not np.any(train):
        raise ValueError("train_mask must select measured rows")
    static, static_names = geometry.static_features()
    rates = np.asarray(retrieval_rate_vectors, dtype=float)
    if rates.shape != (geometry.operation_count, geometry.qubit_count):
        raise ValueError("retrieval_rate_vectors disagree with geometry")
    normalized_rates = rates / max(float(np.max(rates)), 1e-12)
    retrieval_mean = np.mean(normalized_rates, axis=1)
    retrieval_cv = np.std(normalized_rates, axis=1) / np.maximum(retrieval_mean, 1e-12)
    design = np.column_stack((
        np.ones(geometry.operation_count),
        static,
        np.log(np.maximum(retrieval_mean, 1e-12)),
        retrieval_cv,
    ))
    design_names = ("intercept",) + tuple(static_names) + (
        "log_retrieval_operation_mean", "retrieval_operation_cv",
    )
    scale = np.std(design, axis=0)
    scale[scale < 1e-12] = 1.0
    center = np.mean(design, axis=0)
    center[0] = 0.0
    standardized = (design - center) / scale
    standardized[:, 0] = 1.0
    logged = np.stack([
        _soft_normalized_log_share(signal, count_smoothing),
        _soft_normalized_log_share(idler, count_smoothing),
        _soft_normalized_log_share(coin, count_smoothing),
    ], axis=1)
    operation_target = np.median(logged[train], axis=0)
    expected = np.stack([
        standardized @ _ridge_solution(standardized, operation_target[index], response_ridge)
        for index in range(3)
    ])
    expected -= expected.mean(axis=1, keepdims=True)
    if int(response_mode_count) != RESPONSE_MODE_COUNT:
        raise ValueError(f"response_mode_count must remain {RESPONSE_MODE_COUNT} for the V7 schema")
    train_residual_for_modes = logged[train] - expected[None, :, :]
    train_residual_for_modes -= train_residual_for_modes.mean(axis=2, keepdims=True)
    _, _, response_modes = np.linalg.svd(
        train_residual_for_modes.reshape(-1, geometry.operation_count), full_matrices=False,
    )
    response_modes = response_modes[:RESPONSE_MODE_COUNT]

    gsi_coin = coin if gsi_coin_counts is None else np.asarray(gsi_coin_counts, dtype=float)
    if gsi_coin.shape != coin.shape or np.any(gsi_coin < 0.0):
        raise ValueError("gsi_coin_counts must be non-negative and match coin_counts")
    coin_share = (gsi_coin + count_smoothing) / (
        gsi_coin.sum(axis=1, keepdims=True) + count_smoothing * gsi_coin.shape[1]
    )
    accidental = (signal + count_smoothing) * (idler + count_smoothing)
    accidental /= np.maximum(accidental.sum(axis=1, keepdims=True), 1e-12)
    lift = coin_share / np.maximum(accidental, 1e-12)
    contrast_center = np.median(lift[train], axis=0)

    coupling = geometry.coupling_kernel(
        x_length_mhz=float(graph_length_mhz),
        y_length_mhz=float(graph_length_mhz),
        same_local_cross_cycle_weight=0.05,
    )
    coupling += np.eye(geometry.operation_count)
    smoother = coupling / np.maximum(coupling.sum(axis=1, keepdims=True), 1e-12)
    model = PhysicsResponseSurfaceV1(
        system_type=system_type,
        geometry_hash=geometry.geometry_hash,
        operation_design_names=design_names,
        expected_log_share=expected.astype(np.float64),
        operation_contrast_center=contrast_center.astype(np.float64),
        graph_smoother=smoother.astype(np.float64),
        response_modes=response_modes.astype(np.float64),
        retrieval_rate_vectors=rates.astype(np.float64),
        count_smoothing=float(count_smoothing),
        gsi_lower_bound=float(gsi_lower_bound),
        gsi_upper_bound=float(gsi_upper_bound),
        ideal_gsi=float(ideal_gsi),
        training_sample_count=int(np.sum(train)),
    )
    train_residual = logged[train] - expected[None, :, :]
    return model, {
        "schema": "v7_physics_response_surface_fit_v1",
        "training_sample_count": int(np.sum(train)),
        "operation_design_names": list(design_names),
        "train_response_residual_rms": np.sqrt(
            np.mean(np.square(train_residual), axis=(0, 2))
        ).tolist(),
        "positive_response_parameterization": "log_normalized_share",
        "retrieval_field_semantics": "sqrt_count_efficiency",
        "graph_smoothness_enabled": True,
        "response_mode_count": RESPONSE_MODE_COUNT,
        "response_modes_fit_on_training_only": True,
    }


@dataclass(frozen=True)
class ConformalBestFidelityModelV1:
    feature_names: Tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    ridge_alpha: float
    conformal_quantile: float
    coverage_target: float
    training_group_count: int
    tuning_group_count: int
    calibration_group_count: int
    synthetic_prior_weight: float = 0.0
    research_only: bool = True
    promotion_ready: bool = False

    def predict(self, features: Any) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        standardized = (values - self.feature_mean) / self.feature_scale
        design = np.column_stack((np.ones(values.shape[0]), standardized))
        return np.clip(design @ self.coefficients, 0.0, 1.0)

    def upper_bound(self, features: Any) -> np.ndarray:
        return np.clip(self.predict(features) + self.conformal_quantile, 0.0, 1.0)

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for name in ("feature_mean", "feature_scale", "coefficients"):
            payload[name] = np.asarray(payload[name]).tolist()
        payload["schema"] = "v7_conformal_best_fidelity_model_v1"
        return payload


@dataclass(frozen=True)
class StackedConformalBestFidelityModelV1:
    """Target predictor plus validation-gated physics residual correction."""

    target_model: ConformalBestFidelityModelV1
    factor_feature_names: Tuple[str, ...]
    factor_mean: np.ndarray
    factor_scale: np.ndarray
    correction_coefficients: np.ndarray
    correction_strength: float
    correction_ridge_alpha: float
    synthetic_prior_weight: float
    conformal_quantile: float
    coverage_target: float
    target_feature_count: int
    training_group_count: int
    tuning_group_count: int
    calibration_group_count: int
    research_only: bool = True
    promotion_ready: bool = False

    def predict(self, features: Any) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        target = values[:, :self.target_feature_count]
        factors = values[:, self.target_feature_count:]
        if factors.shape[1] != len(self.factor_feature_names):
            raise ValueError("stacked best-fidelity feature shape disagrees with model")
        standardized = (factors - self.factor_mean) / self.factor_scale
        design = np.column_stack((np.ones(factors.shape[0]), standardized))
        correction = design @ self.correction_coefficients
        return np.clip(
            self.target_model.predict(target) + self.correction_strength * correction,
            0.0,
            1.0,
        )

    def upper_bound(self, features: Any) -> np.ndarray:
        return np.clip(self.predict(features) + self.conformal_quantile, 0.0, 1.0)

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["target_model"] = self.target_model.to_json_dict()
        for name in ("factor_mean", "factor_scale", "correction_coefficients"):
            payload[name] = np.asarray(payload[name]).tolist()
        payload["schema"] = "v7_stacked_conformal_best_fidelity_model_v1"
        return payload


def fit_conformal_best_fidelity_model(
    features: Any,
    labels: Any,
    train_mask: Any,
    tune_mask: Any,
    calibration_mask: Any,
    feature_names: Sequence[str],
    *,
    ridge_alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
    coverage: float = 0.90,
    coefficient_prior: Optional[Any] = None,
    prior_weights: Sequence[float] = (0.0,),
) -> Tuple[ConformalBestFidelityModelV1, Dict[str, Any]]:
    """Fit a point predictor and independent one-sided conformal upper bound."""

    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float).reshape(-1)
    train = np.asarray(train_mask, dtype=bool).reshape(-1)
    tune = np.asarray(tune_mask, dtype=bool).reshape(-1)
    calibrate = np.asarray(calibration_mask, dtype=bool).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.size or len(feature_names) != x.shape[1]:
        raise ValueError("best-fidelity feature shapes disagree")
    if any(mask.shape != (y.size,) or not np.any(mask) for mask in (train, tune, calibrate)):
        raise ValueError("train/tune/calibration masks must each select groups")
    if np.any(train & tune) or np.any(train & calibrate) or np.any(tune & calibrate):
        raise ValueError("train/tune/calibration masks must be disjoint")
    if not 0.5 < float(coverage) < 1.0:
        raise ValueError("coverage must lie in (0.5, 1)")
    mean = np.mean(x[train], axis=0)
    scale = np.std(x[train], axis=0)
    scale[scale < 1e-9] = 1.0
    standardized = (x - mean) / scale
    design = np.column_stack((np.ones(x.shape[0]), standardized))
    prior = None if coefficient_prior is None else np.asarray(coefficient_prior, dtype=float)
    candidates = []
    for alpha_value in ridge_alphas:
        for prior_weight_value in prior_weights:
            alpha = float(alpha_value)
            prior_weight = float(prior_weight_value)
            coefficients = _ridge_solution(
                design[train], y[train], alpha,
                coefficient_prior=prior, prior_weight=prior_weight,
            )
            tune_prediction = np.clip(design[tune] @ coefficients, 0.0, 1.0)
            tune_mae = float(np.mean(np.abs(tune_prediction - y[tune])))
            candidates.append((tune_mae, alpha, prior_weight, coefficients))
    tune_mae, selected_alpha, selected_prior_weight, coefficients = min(
        candidates, key=lambda item: (item[0], item[1], item[2]),
    )
    calibration_prediction = np.clip(design[calibrate] @ coefficients, 0.0, 1.0)
    conformity = y[calibrate] - calibration_prediction
    level = min(
        1.0,
        np.ceil((conformity.size + 1) * float(coverage)) / conformity.size,
    )
    conformal_quantile = float(np.quantile(conformity, level, method="higher"))
    model = ConformalBestFidelityModelV1(
        feature_names=tuple(str(name) for name in feature_names),
        feature_mean=mean.astype(np.float64),
        feature_scale=scale.astype(np.float64),
        coefficients=coefficients.astype(np.float64),
        ridge_alpha=selected_alpha,
        conformal_quantile=conformal_quantile,
        coverage_target=float(coverage),
        training_group_count=int(np.sum(train)),
        tuning_group_count=int(np.sum(tune)),
        calibration_group_count=int(np.sum(calibrate)),
        synthetic_prior_weight=selected_prior_weight,
    )
    return model, {
        "schema": "v7_conformal_best_fidelity_fit_v1",
        "selected_ridge_alpha": selected_alpha,
        "selected_prior_weight": selected_prior_weight,
        "tuning_mae": tune_mae,
        "conformal_quantile": conformal_quantile,
        "coverage_target": float(coverage),
        "finite_sample_quantile_level": float(level),
        "candidate_table": [
            {"tuning_mae": item[0], "ridge_alpha": item[1], "prior_weight": item[2]}
            for item in candidates
        ],
    }


def fit_grouped_conformal_best_fidelity_model(
    features: Any,
    labels: Any,
    group_ids: Any,
    train_mask: Any,
    tune_mask: Any,
    calibration_mask: Any,
    feature_names: Sequence[str],
    *,
    ridge_alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
    coverage: float = 0.90,
) -> Tuple[ConformalBestFidelityModelV1, Dict[str, Any]]:
    """Fit sample fidelity while selecting/calibrating on grouped best fidelity."""

    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float).reshape(-1)
    groups = np.asarray(group_ids).astype(str)
    train = np.asarray(train_mask, dtype=bool)
    tune = np.asarray(tune_mask, dtype=bool)
    calibrate = np.asarray(calibration_mask, dtype=bool)
    if x.ndim != 2 or x.shape[0] != y.size or groups.shape != (y.size,):
        raise ValueError("grouped best-fidelity feature shapes disagree")
    if any(mask.shape != (y.size,) or not np.any(mask) for mask in (train, tune, calibrate)):
        raise ValueError("train/tune/calibration masks must each select samples")
    mean = np.mean(x[train], axis=0)
    scale = np.std(x[train], axis=0)
    scale[scale < 1e-9] = 1.0
    design = np.column_stack((np.ones(x.shape[0]), (x - mean) / scale))
    weights = _equal_group_weights(groups, train)
    weighted_design = design[train] * np.sqrt(weights[train])[:, None]
    weighted_labels = y[train] * np.sqrt(weights[train])
    candidates = []
    for alpha_value in ridge_alphas:
        coefficients = _ridge_solution(weighted_design, weighted_labels, float(alpha_value))
        prediction = np.clip(design @ coefficients, 0.0, 1.0)
        tune_prediction, tune_label, _ = _group_max_values(prediction, y, groups, tune)
        candidates.append((
            float(np.mean(np.abs(tune_prediction - tune_label))),
            float(alpha_value),
            coefficients,
        ))
    tune_mae, selected_alpha, coefficients = min(
        candidates, key=lambda item: (item[0], item[1]),
    )
    prediction = np.clip(design @ coefficients, 0.0, 1.0)
    calibration_prediction, calibration_label, _ = _group_max_values(
        prediction, y, groups, calibrate,
    )
    conformity = calibration_label - calibration_prediction
    level = min(1.0, np.ceil((conformity.size + 1) * float(coverage)) / conformity.size)
    conformal_quantile = float(np.quantile(conformity, level, method="higher"))
    model = ConformalBestFidelityModelV1(
        feature_names=tuple(str(name) for name in feature_names),
        feature_mean=mean.astype(np.float64),
        feature_scale=scale.astype(np.float64),
        coefficients=coefficients.astype(np.float64),
        ridge_alpha=selected_alpha,
        conformal_quantile=conformal_quantile,
        coverage_target=float(coverage),
        training_group_count=int(np.unique(groups[train]).size),
        tuning_group_count=int(np.unique(groups[tune]).size),
        calibration_group_count=int(np.unique(groups[calibrate]).size),
    )
    return model, {
        "schema": "v7_grouped_conformal_best_fidelity_fit_v1",
        "selected_ridge_alpha": selected_alpha,
        "tuning_group_mae": tune_mae,
        "conformal_quantile": conformal_quantile,
        "coverage_target": float(coverage),
        "finite_sample_quantile_level": float(level),
        "equal_total_weight_per_training_group": True,
        "candidate_table": [
            {"tuning_group_mae": item[0], "ridge_alpha": item[1]}
            for item in candidates
        ],
    }


def fit_stacked_conformal_best_fidelity_model(
    target_model: ConformalBestFidelityModelV1,
    target_features: Any,
    factor_features: Any,
    labels: Any,
    train_mask: Any,
    tune_mask: Any,
    calibration_mask: Any,
    factor_feature_names: Sequence[str],
    *,
    ridge_alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
    correction_strengths: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
    coverage: float = 0.90,
    coefficient_prior: Optional[Any] = None,
    prior_weights: Sequence[float] = (0.0,),
) -> Tuple[StackedConformalBestFidelityModelV1, Dict[str, Any]]:
    """Fit only the fidelity residual left by an immutable target model."""

    target = np.asarray(target_features, dtype=float)
    factors = np.asarray(factor_features, dtype=float)
    y = np.asarray(labels, dtype=float).reshape(-1)
    train = np.asarray(train_mask, dtype=bool).reshape(-1)
    tune = np.asarray(tune_mask, dtype=bool).reshape(-1)
    calibrate = np.asarray(calibration_mask, dtype=bool).reshape(-1)
    if target.ndim != 2 or factors.ndim != 2 or target.shape[0] != y.size or factors.shape[0] != y.size:
        raise ValueError("stacked best-fidelity feature shapes disagree")
    if factors.shape[1] != len(factor_feature_names):
        raise ValueError("factor feature names disagree with factor matrix")
    if any(mask.shape != (y.size,) or not np.any(mask) for mask in (train, tune, calibrate)):
        raise ValueError("train/tune/calibration masks must each select groups")
    mean = np.mean(factors[train], axis=0)
    scale = np.std(factors[train], axis=0)
    scale[scale < 1e-9] = 1.0
    standardized = (factors - mean) / scale
    design = np.column_stack((np.ones(factors.shape[0]), standardized))
    base_prediction = target_model.predict(target)
    residual_label = y - base_prediction
    prior = None if coefficient_prior is None else np.asarray(coefficient_prior, dtype=float)
    candidates = []
    for alpha_value in ridge_alphas:
        for prior_weight_value in prior_weights:
            coefficients = _ridge_solution(
                design[train], residual_label[train], float(alpha_value),
                coefficient_prior=prior, prior_weight=float(prior_weight_value),
            )
            raw_correction = design[tune] @ coefficients
            for strength_value in correction_strengths:
                strength = float(strength_value)
                prediction = np.clip(base_prediction[tune] + strength * raw_correction, 0.0, 1.0)
                mae = float(np.mean(np.abs(prediction - y[tune])))
                candidates.append((
                    mae, strength, float(alpha_value), float(prior_weight_value), coefficients,
                ))
    tune_mae, selected_strength, selected_alpha, selected_prior_weight, coefficients = min(
        candidates, key=lambda item: (item[0], item[1], item[2], item[3]),
    )
    calibration_prediction = np.clip(
        base_prediction[calibrate]
        + selected_strength * (design[calibrate] @ coefficients),
        0.0,
        1.0,
    )
    conformity = y[calibrate] - calibration_prediction
    level = min(1.0, np.ceil((conformity.size + 1) * float(coverage)) / conformity.size)
    conformal_quantile = float(np.quantile(conformity, level, method="higher"))
    model = StackedConformalBestFidelityModelV1(
        target_model=target_model,
        factor_feature_names=tuple(str(name) for name in factor_feature_names),
        factor_mean=mean.astype(np.float64),
        factor_scale=scale.astype(np.float64),
        correction_coefficients=coefficients.astype(np.float64),
        correction_strength=selected_strength,
        correction_ridge_alpha=selected_alpha,
        synthetic_prior_weight=selected_prior_weight,
        conformal_quantile=conformal_quantile,
        coverage_target=float(coverage),
        target_feature_count=target.shape[1],
        training_group_count=int(np.sum(train)),
        tuning_group_count=int(np.sum(tune)),
        calibration_group_count=int(np.sum(calibrate)),
    )
    return model, {
        "schema": "v7_stacked_conformal_best_fidelity_fit_v1",
        "selected_correction_strength": selected_strength,
        "selected_ridge_alpha": selected_alpha,
        "selected_prior_weight": selected_prior_weight,
        "tuning_mae": tune_mae,
        "target_only_tuning_mae": float(np.mean(np.abs(base_prediction[tune] - y[tune]))),
        "conformal_quantile": conformal_quantile,
        "coverage_target": float(coverage),
        "finite_sample_quantile_level": float(level),
        "candidate_table": [
            {
                "tuning_mae": item[0],
                "correction_strength": item[1],
                "ridge_alpha": item[2],
                "prior_weight": item[3],
            }
            for item in candidates
        ],
    }


def fit_grouped_stacked_best_fidelity_model(
    target_model: ConformalBestFidelityModelV1,
    target_features: Any,
    factor_features: Any,
    labels: Any,
    group_ids: Any,
    train_mask: Any,
    tune_mask: Any,
    calibration_mask: Any,
    factor_feature_names: Sequence[str],
    *,
    ridge_alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
    correction_strengths: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
    coverage: float = 0.90,
    coefficient_prior: Optional[Any] = None,
    prior_weights: Sequence[float] = (0.0,),
) -> Tuple[StackedConformalBestFidelityModelV1, Dict[str, Any]]:
    """Fit sample-level physical residuals and gate them on grouped bestF."""

    target = np.asarray(target_features, dtype=float)
    factors = np.asarray(factor_features, dtype=float)
    y = np.asarray(labels, dtype=float).reshape(-1)
    groups = np.asarray(group_ids).astype(str)
    train = np.asarray(train_mask, dtype=bool)
    tune = np.asarray(tune_mask, dtype=bool)
    calibrate = np.asarray(calibration_mask, dtype=bool)
    if target.ndim != 2 or factors.ndim != 2 or target.shape[0] != y.size or factors.shape[0] != y.size:
        raise ValueError("grouped stacked feature shapes disagree")
    mean = np.mean(factors[train], axis=0)
    scale = np.std(factors[train], axis=0)
    scale[scale < 1e-9] = 1.0
    design = np.column_stack((np.ones(factors.shape[0]), (factors - mean) / scale))
    base_prediction = target_model.predict(target)
    residual_label = y - base_prediction
    weights = _equal_group_weights(groups, train)
    weighted_design = design[train] * np.sqrt(weights[train])[:, None]
    weighted_residual = residual_label[train] * np.sqrt(weights[train])
    prior = None if coefficient_prior is None else np.asarray(coefficient_prior, dtype=float)
    candidates = []
    for alpha_value in ridge_alphas:
        for prior_weight_value in prior_weights:
            coefficients = _ridge_solution(
                weighted_design,
                weighted_residual,
                float(alpha_value),
                coefficient_prior=prior,
                prior_weight=float(prior_weight_value),
            )
            raw_correction = design @ coefficients
            for strength_value in correction_strengths:
                prediction = np.clip(
                    base_prediction + float(strength_value) * raw_correction, 0.0, 1.0,
                )
                tune_prediction, tune_label, _ = _group_max_values(prediction, y, groups, tune)
                candidates.append((
                    float(np.mean(np.abs(tune_prediction - tune_label))),
                    float(strength_value),
                    float(alpha_value),
                    float(prior_weight_value),
                    coefficients,
                ))
    tune_mae, selected_strength, selected_alpha, selected_prior_weight, coefficients = min(
        candidates, key=lambda item: (item[0], item[1], item[2], item[3]),
    )
    prediction = np.clip(
        base_prediction + selected_strength * (design @ coefficients), 0.0, 1.0,
    )
    calibration_prediction, calibration_label, _ = _group_max_values(
        prediction, y, groups, calibrate,
    )
    conformity = calibration_label - calibration_prediction
    level = min(1.0, np.ceil((conformity.size + 1) * float(coverage)) / conformity.size)
    conformal_quantile = float(np.quantile(conformity, level, method="higher"))
    model = StackedConformalBestFidelityModelV1(
        target_model=target_model,
        factor_feature_names=tuple(str(name) for name in factor_feature_names),
        factor_mean=mean.astype(np.float64),
        factor_scale=scale.astype(np.float64),
        correction_coefficients=coefficients.astype(np.float64),
        correction_strength=selected_strength,
        correction_ridge_alpha=selected_alpha,
        synthetic_prior_weight=selected_prior_weight,
        conformal_quantile=conformal_quantile,
        coverage_target=float(coverage),
        target_feature_count=target.shape[1],
        training_group_count=int(np.unique(groups[train]).size),
        tuning_group_count=int(np.unique(groups[tune]).size),
        calibration_group_count=int(np.unique(groups[calibrate]).size),
    )
    base_tune, tune_label, _ = _group_max_values(base_prediction, y, groups, tune)
    return model, {
        "schema": "v7_grouped_stacked_best_fidelity_fit_v1",
        "selected_correction_strength": selected_strength,
        "selected_ridge_alpha": selected_alpha,
        "selected_prior_weight": selected_prior_weight,
        "tuning_group_mae": tune_mae,
        "target_only_tuning_group_mae": float(np.mean(np.abs(base_tune - tune_label))),
        "conformal_quantile": conformal_quantile,
        "coverage_target": float(coverage),
        "finite_sample_quantile_level": float(level),
        "equal_total_weight_per_training_group": True,
        "candidate_table": [
            {
                "tuning_group_mae": item[0],
                "correction_strength": item[1],
                "ridge_alpha": item[2],
                "prior_weight": item[3],
            }
            for item in candidates
        ],
    }


def evaluate_best_fidelity_bounds(
    model: ConformalBestFidelityModelV1,
    features: Any,
    labels: Any,
    physics_hypothesis_ceiling: Any,
) -> Dict[str, Any]:
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float).reshape(-1)
    physics = np.asarray(physics_hypothesis_ceiling, dtype=float).reshape(-1)
    prediction = model.predict(x)
    upper = model.upper_bound(x)
    if y.shape != prediction.shape or physics.shape != y.shape:
        raise ValueError("best-fidelity evaluation shapes disagree")
    return {
        "group_count": int(y.size),
        "best_fidelity_mean": float(np.mean(y)),
        "best_fidelity_max": float(np.max(y)),
        "point_mae": float(np.mean(np.abs(prediction - y))),
        "point_rmse": float(np.sqrt(np.mean(np.square(prediction - y)))),
        "statistical_upper_coverage": float(np.mean(y <= upper + 1e-12)),
        "statistical_upper_mean": float(np.mean(upper)),
        "statistical_upper_excess_mean": float(np.mean(upper - y)),
        "physics_hypothesis_coverage": float(np.mean(y <= physics + 1e-12)),
        "physics_hypothesis_ceiling_mean": float(np.mean(physics)),
        "physics_hypothesis_excess_mean": float(np.mean(physics - y)),
        "physics_hypothesis_invalid_for_any_group": bool(np.any(y > physics + 1e-12)),
    }


def evaluate_grouped_best_fidelity_bounds(
    model: Any,
    features: Any,
    labels: Any,
    group_ids: Any,
    sample_mask: Any,
    physics_hypothesis_ceiling: Any,
) -> Dict[str, Any]:
    values = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float).reshape(-1)
    groups = np.asarray(group_ids).astype(str)
    mask = np.asarray(sample_mask, dtype=bool)
    physics = np.asarray(physics_hypothesis_ceiling, dtype=float).reshape(-1)
    prediction = model.predict(values)
    predicted_best, observed_best, unique_groups = _group_max_values(
        prediction, y, groups, mask,
    )
    physics_best = np.asarray([
        np.max(physics[mask & (groups == group_id)]) for group_id in unique_groups
    ])
    upper = np.clip(predicted_best + float(model.conformal_quantile), 0.0, 1.0)
    return {
        "group_count": int(observed_best.size),
        "best_fidelity_mean": float(np.mean(observed_best)),
        "best_fidelity_max": float(np.max(observed_best)),
        "point_mae": float(np.mean(np.abs(predicted_best - observed_best))),
        "point_rmse": float(np.sqrt(np.mean(np.square(predicted_best - observed_best)))),
        "statistical_upper_coverage": float(np.mean(observed_best <= upper + 1e-12)),
        "statistical_upper_mean": float(np.mean(upper)),
        "statistical_upper_excess_mean": float(np.mean(upper - observed_best)),
        "physics_hypothesis_coverage": float(np.mean(observed_best <= physics_best + 1e-12)),
        "physics_hypothesis_ceiling_mean": float(np.mean(physics_best)),
        "physics_hypothesis_excess_mean": float(np.mean(physics_best - observed_best)),
        "physics_hypothesis_invalid_for_any_group": bool(
            np.any(observed_best > physics_best + 1e-12)
        ),
    }


__all__ = [
    "ConformalBestFidelityModelV1",
    "PHYSICS_CEILING_SCHEMA",
    "PHYSICS_FACTOR_NAMES",
    "PhysicsResponseSurfaceV1",
    "StackedConformalBestFidelityModelV1",
    "evaluate_best_fidelity_bounds",
    "evaluate_grouped_best_fidelity_bounds",
    "fit_conformal_best_fidelity_model",
    "fit_grouped_conformal_best_fidelity_model",
    "fit_grouped_stacked_best_fidelity_model",
    "fit_physics_response_surface",
    "fit_stacked_conformal_best_fidelity_model",
    "load_retrieval_rate_vectors",
    "pure_target_fidelity",
    "retrieval_ceiling_features",
]
