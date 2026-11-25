"""Joint NB2 coherent/accidental count model for V7 research.

The model replaces deterministic count subtraction with one positive forward
map.  Its operation-local mean is the sum of a coherent target-conditioned
component and a singles-conditioned accidental component.  Global multi-click
counts may condition the sample-level coherent fraction and total rate, but are
never expanded into operation-local pseudo-counts.

The coherent response is a bilinear target-by-operation law.  Operation
features are restricted to nominal AOD frequency geometry, tomography basis,
retrieval imbalance, and smooth graph modes.  This preserves phase-sensitive
target structure while avoiding an unrestricted per-target lookup table.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ml_recon.physics_identification.hierarchical_nb2 import nb2_nll

from .geometry import OperationFrequencyGeometryV1


JOINT_NB2_SCHEMA = "v7_joint_nb2_mixture_v1"
JOINT_PHYSICS_FACTOR_NAMES = (
    "joint_coherent_fraction",
    "joint_noise_fraction",
    "coherent_accidental_total_variation",
    "effective_coherent_contrast",
    "coherent_delta_rms",
    "nb2_pearson_residual_rms",
    "nb2_pearson_residual_bias",
    "nb2_deviance_per_operation",
    "nb2_fisher_log_total",
    "absolute_log_total_rate_error",
)
SAMPLE_CONTEXT_NAMES = (
    "log_signal_total",
    "log_idler_total",
    "log_signal_idler_ratio",
    "log_singles_geomean",
    "log_exposure_proxy",
    "multi_click_per_singles_geomean",
)


@dataclass(frozen=True)
class V7JointNB2Arrays:
    coin_counts: np.ndarray
    signal_counts: np.ndarray
    idler_counts: np.ndarray
    target_features: np.ndarray
    multi_click_counts: np.ndarray
    exposure_proxy: np.ndarray
    sample_weights: np.ndarray

    def validate(self, operation_count: int) -> None:
        coin = np.asarray(self.coin_counts, dtype=float)
        signal = np.asarray(self.signal_counts, dtype=float)
        idler = np.asarray(self.idler_counts, dtype=float)
        if coin.ndim != 2 or coin.shape != signal.shape or coin.shape != idler.shape:
            raise ValueError("coin/signal/idler counts must have the same two-dimensional shape")
        if coin.shape[1] != int(operation_count):
            raise ValueError("count arrays disagree with operation geometry")
        if any(np.any(~np.isfinite(value)) or np.any(value < 0.0) for value in (coin, signal, idler)):
            raise ValueError("count arrays must be finite and non-negative")
        target = np.asarray(self.target_features, dtype=float)
        multi = np.asarray(self.multi_click_counts, dtype=float)
        exposure = np.asarray(self.exposure_proxy, dtype=float).reshape(-1)
        weights = np.asarray(self.sample_weights, dtype=float).reshape(-1)
        if target.ndim != 2 or target.shape[0] != coin.shape[0]:
            raise ValueError("target_features must have one row per sample")
        if multi.ndim != 2 or multi.shape[0] != coin.shape[0]:
            raise ValueError("multi_click_counts must have one row per sample")
        if exposure.shape != (coin.shape[0],) or weights.shape != exposure.shape:
            raise ValueError("exposure_proxy/sample_weights must have one value per sample")
        if np.any(multi < 0.0) or np.any(exposure < 0.0) or np.any(weights <= 0.0):
            raise ValueError("multi-click, exposure, and sample weights contain invalid values")


def sample_context_features(arrays: V7JointNB2Arrays) -> np.ndarray:
    signal_total = np.sum(np.asarray(arrays.signal_counts, dtype=float), axis=1)
    idler_total = np.sum(np.asarray(arrays.idler_counts, dtype=float), axis=1)
    singles_geomean = np.sqrt(np.maximum(signal_total * idler_total, 1.0))
    multi_total = np.sum(np.asarray(arrays.multi_click_counts, dtype=float), axis=1)
    return np.column_stack((
        np.log1p(signal_total),
        np.log1p(idler_total),
        np.log((signal_total + 1.0) / (idler_total + 1.0)),
        np.log1p(singles_geomean),
        np.log1p(np.asarray(arrays.exposure_proxy, dtype=float)),
        multi_total / (singles_geomean + 1.0),
    )).astype(np.float64)


def build_joint_operation_design(
    geometry: OperationFrequencyGeometryV1,
    retrieval_rate_vectors: Any,
    *,
    graph_mode_count: int,
    graph_length_mhz: float,
) -> Tuple[np.ndarray, Tuple[str, ...], Dict[str, Any]]:
    """Return a deterministic smooth AOD/retrieval operation dictionary."""

    static, static_names = geometry.static_features()
    rates = np.asarray(retrieval_rate_vectors, dtype=float)
    if rates.shape != (geometry.operation_count, geometry.qubit_count):
        raise ValueError("retrieval_rate_vectors disagree with geometry")
    normalized_rates = rates / max(float(np.max(rates)), 1e-12)
    retrieval_mean = np.mean(normalized_rates, axis=1)
    retrieval_cv = np.std(normalized_rates, axis=1) / np.maximum(retrieval_mean, 1e-12)
    base = np.column_stack((
        static,
        np.log(np.maximum(retrieval_mean, 1e-12)),
        retrieval_cv,
    ))
    names = list(static_names) + [
        "log_retrieval_operation_mean",
        "retrieval_operation_cv",
    ]
    scale = np.std(base, axis=0)
    keep = scale > 1e-10
    base = base[:, keep]
    names = [name for name, selected in zip(names, keep) if selected]
    base = (base - np.mean(base, axis=0)) / np.std(base, axis=0)

    requested_modes = max(0, min(int(graph_mode_count), geometry.operation_count - 1))
    if requested_modes:
        kernel = geometry.coupling_kernel(
            x_length_mhz=float(graph_length_mhz),
            y_length_mhz=float(graph_length_mhz),
            same_local_cross_cycle_weight=0.05,
        )
        degree = np.maximum(np.sum(kernel, axis=1), 1e-12)
        normalized_kernel = kernel / np.sqrt(degree[:, None] * degree[None, :])
        eigenvalues, eigenvectors = np.linalg.eigh(normalized_kernel)
        order = np.argsort(eigenvalues)[::-1]
        graph_modes = eigenvectors[:, order[:requested_modes]].copy()
        for mode_index in range(graph_modes.shape[1]):
            pivot = int(np.argmax(np.abs(graph_modes[:, mode_index])))
            if graph_modes[pivot, mode_index] < 0.0:
                graph_modes[:, mode_index] *= -1.0
        graph_modes -= np.mean(graph_modes, axis=0)
        graph_modes /= np.maximum(np.std(graph_modes, axis=0), 1e-12)
        base = np.column_stack((base, graph_modes))
        names.extend(f"frequency_graph_mode_{index + 1}" for index in range(requested_modes))
    return base.astype(np.float64), tuple(names), {
        "operation_feature_count": int(base.shape[1]),
        "graph_mode_count": requested_modes,
        "retrieval_field_semantics": "sqrt_count_efficiency_in_physical_interpretation",
        "historical_command_invented": False,
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    centered = logits - np.max(logits, axis=1, keepdims=True)
    probability = np.exp(np.clip(centered, -60.0, 60.0))
    return probability / np.maximum(np.sum(probability, axis=1, keepdims=True), 1e-12)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _conditional_nll_per_click(counts: np.ndarray, probability: np.ndarray) -> np.ndarray:
    predicted = np.maximum(np.asarray(probability, dtype=float), 1e-12)
    predicted /= np.sum(predicted, axis=1, keepdims=True)
    values = np.asarray(counts, dtype=float)
    return -np.sum(values * np.log(predicted), axis=1) / np.maximum(np.sum(values, axis=1), 1.0)


@dataclass(frozen=True)
class V7JointNB2Model:
    system_type: str
    geometry_hash: str
    target_feature_names: Tuple[str, ...]
    operation_feature_names: Tuple[str, ...]
    operation_design: np.ndarray
    target_feature_mean: np.ndarray
    target_feature_scale: np.ndarray
    sample_feature_mean: np.ndarray
    sample_feature_scale: np.ndarray
    coherent_coefficients: np.ndarray
    accidental_coefficients: np.ndarray
    total_rate_coefficients: np.ndarray
    coherent_fraction_coefficients: np.ndarray
    coherent_fraction_lower: float
    coherent_fraction_upper: float
    nb2_dispersion: float
    dark_fraction: float
    dark_supervision_available: bool
    count_smoothing: float
    observed_q_prior_strength: float
    q_refinement_grid_size: int
    training_sample_count: int
    synthetic_pretrained: bool
    synthetic_prior_weight: float
    research_only: bool = True
    promotion_ready: bool = False
    hardware_action_authorized: bool = False

    def _designs(self, arrays: V7JointNB2Arrays) -> Tuple[np.ndarray, np.ndarray]:
        target = (np.asarray(arrays.target_features, dtype=float) - self.target_feature_mean) / self.target_feature_scale
        target = np.column_stack((np.ones(target.shape[0]), target))
        sample = (sample_context_features(arrays) - self.sample_feature_mean) / self.sample_feature_scale
        sample = np.column_stack((np.ones(sample.shape[0]), sample))
        return target, sample

    def components(self, arrays: V7JointNB2Arrays) -> Dict[str, np.ndarray]:
        arrays.validate(self.operation_design.shape[0])
        target, sample = self._designs(arrays)
        accidental_seed = (
            np.asarray(arrays.signal_counts, dtype=float) + self.count_smoothing
        ) * (
            np.asarray(arrays.idler_counts, dtype=float) + self.count_smoothing
        )
        accidental_logits = np.log(np.maximum(accidental_seed, 1e-12))
        accidental_logits += self.operation_design @ self.accidental_coefficients
        accidental_probability = _softmax(accidental_logits)
        coherent_delta = (target @ self.coherent_coefficients) @ self.operation_design.T
        coherent_delta -= np.mean(coherent_delta, axis=1, keepdims=True)
        coherent_probability = _softmax(accidental_logits + np.clip(coherent_delta, -8.0, 8.0))
        unit_interval = _sigmoid(sample @ self.coherent_fraction_coefficients)
        coherent_fraction = self.coherent_fraction_lower + (
            self.coherent_fraction_upper - self.coherent_fraction_lower
        ) * unit_interval
        dark_probability = np.full_like(accidental_probability, 1.0 / accidental_probability.shape[1])
        active_fraction = 1.0 - float(self.dark_fraction)
        probability = active_fraction * (
            coherent_fraction[:, None] * coherent_probability
            + (1.0 - coherent_fraction[:, None]) * accidental_probability
        ) + float(self.dark_fraction) * dark_probability
        probability /= np.sum(probability, axis=1, keepdims=True)
        total_mean = np.exp(np.clip(sample @ self.total_rate_coefficients, -5.0, 20.0))
        expected_counts = total_mean[:, None] * probability
        return {
            "probability": probability,
            "expected_counts": expected_counts,
            "expected_total": total_mean,
            "coherent_probability": coherent_probability,
            "accidental_probability": accidental_probability,
            "dark_probability": dark_probability,
            "coherent_fraction": coherent_fraction,
            "noise_fraction": 1.0 - coherent_fraction,
            "coherent_delta": coherent_delta,
        }

    def predict_hidden_probability(
        self,
        arrays: V7JointNB2Arrays,
        observed_coin_counts: Any,
        observed_mask: Any,
        *,
        refine_coherent_fraction: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        mask = np.asarray(observed_mask, dtype=bool).reshape(-1)
        observed = np.asarray(observed_coin_counts, dtype=float)
        if mask.shape != (self.operation_design.shape[0],) or not np.any(mask) or np.all(mask):
            raise ValueError("observed_mask must select a strict operation subset")
        if observed.shape != (len(arrays.coin_counts), int(np.sum(mask))):
            raise ValueError("observed_coin_counts disagree with arrays/observed_mask")
        components = self.components(arrays)
        coherent = components["coherent_probability"]
        accidental = components["accidental_probability"]
        prior_q = components["coherent_fraction"]
        selected_q = prior_q.copy()
        if refine_coherent_fraction:
            grid = np.linspace(
                self.coherent_fraction_lower,
                self.coherent_fraction_upper,
                max(3, int(self.q_refinement_grid_size)),
            )
            candidate_loss = []
            prior_logit = np.log(
                np.maximum(prior_q - self.coherent_fraction_lower, 1e-8)
                / np.maximum(self.coherent_fraction_upper - prior_q, 1e-8)
            )
            for candidate in grid:
                probability = (
                    candidate * coherent[:, mask]
                    + (1.0 - candidate) * accidental[:, mask]
                )
                loss = _conditional_nll_per_click(observed, probability)
                candidate_unit = (
                    candidate - self.coherent_fraction_lower
                ) / max(self.coherent_fraction_upper - self.coherent_fraction_lower, 1e-12)
                candidate_logit = np.log(
                    max(candidate_unit, 1e-8) / max(1.0 - candidate_unit, 1e-8)
                )
                loss += self.observed_q_prior_strength * np.square(
                    candidate_logit - prior_logit
                ) / np.maximum(np.sum(observed, axis=1), 1.0)
                candidate_loss.append(loss)
            loss_matrix = np.stack(candidate_loss, axis=1)
            selected_q = grid[np.argmin(loss_matrix, axis=1)]
        hidden_probability = (
            selected_q[:, None] * coherent[:, ~mask]
            + (1.0 - selected_q[:, None]) * accidental[:, ~mask]
        )
        hidden_probability = (
            (1.0 - self.dark_fraction) * hidden_probability
            + self.dark_fraction / hidden_probability.shape[1]
        )
        hidden_probability /= np.sum(hidden_probability, axis=1, keepdims=True)
        return hidden_probability, {
            "prior_coherent_fraction": prior_q,
            "selected_coherent_fraction": selected_q,
            "observed_coin_used_only_for_global_q_refinement": np.ones(prior_q.shape, dtype=bool),
        }

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for name in (
            "operation_design", "target_feature_mean", "target_feature_scale",
            "sample_feature_mean", "sample_feature_scale", "coherent_coefficients",
            "accidental_coefficients", "total_rate_coefficients",
            "coherent_fraction_coefficients",
        ):
            payload[name] = np.asarray(payload[name]).tolist()
        payload["schema"] = JOINT_NB2_SCHEMA
        payload["count_mean_formula"] = (
            "expected_total * ((1-dark)*(q*coherent + (1-q)*accidental) + dark*uniform)"
        )
        payload["global_multi_click_scope_preserved"] = True
        payload["production_subtraction_allowed"] = False
        return payload

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "V7JointNB2Model":
        values = dict(payload)
        if values.pop("schema", None) != JOINT_NB2_SCHEMA:
            raise ValueError("unsupported V7 joint NB2 model schema")
        for metadata_name in (
            "count_mean_formula", "global_multi_click_scope_preserved",
            "production_subtraction_allowed",
        ):
            values.pop(metadata_name, None)
        values["target_feature_names"] = tuple(values["target_feature_names"])
        values["operation_feature_names"] = tuple(values["operation_feature_names"])
        for name in (
            "operation_design", "target_feature_mean", "target_feature_scale",
            "sample_feature_mean", "sample_feature_scale", "coherent_coefficients",
            "accidental_coefficients", "total_rate_coefficients",
            "coherent_fraction_coefficients",
        ):
            values[name] = np.asarray(values[name], dtype=np.float64)
        return cls(**values)


def joint_physics_factors(
    model: V7JointNB2Model,
    arrays: V7JointNB2Arrays,
) -> Tuple[np.ndarray, Tuple[str, ...], Dict[str, np.ndarray]]:
    """Return identifiable joint-count factors for fidelity-ceiling analysis."""

    components = model.components(arrays)
    observed = np.asarray(arrays.coin_counts, dtype=float)
    mean = np.maximum(components["expected_counts"], 1e-12)
    variance = mean + float(model.nb2_dispersion) * np.square(mean)
    pearson = (observed - mean) / np.sqrt(np.maximum(variance, 1e-12))
    if model.nb2_dispersion <= 1e-10:
        deviance = 2.0 * (
            observed * np.log(np.maximum(observed, 1e-12) / mean) - observed + mean
        )
    else:
        alpha = float(model.nb2_dispersion)
        size = 1.0 / alpha
        deviance = 2.0 * (
            observed * np.log(np.maximum(observed, 1e-12) / mean)
            - (observed + size) * np.log((observed + size) / (mean + size))
        )
    deviance = np.maximum(deviance, 0.0)
    total_variation = 0.5 * np.sum(np.abs(
        components["coherent_probability"] - components["accidental_probability"]
    ), axis=1)
    coherent_fraction = components["coherent_fraction"]
    observed_total = np.sum(observed, axis=1)
    predicted_total = components["expected_total"]
    fisher = np.sum(mean / (1.0 + float(model.nb2_dispersion) * mean), axis=1)
    factors = np.column_stack((
        coherent_fraction,
        1.0 - coherent_fraction,
        total_variation,
        coherent_fraction * total_variation,
        np.sqrt(np.mean(np.square(components["coherent_delta"]), axis=1)),
        np.sqrt(np.mean(np.square(pearson), axis=1)),
        np.mean(pearson, axis=1),
        np.mean(deviance, axis=1),
        fisher,
        np.abs(np.log((observed_total + 1.0) / (predicted_total + 1.0))),
    ))
    return factors.astype(np.float64), JOINT_PHYSICS_FACTOR_NAMES, {
        "coherent_fraction": coherent_fraction.astype(np.float64),
        "component_total_variation": total_variation.astype(np.float64),
        "effective_coherent_contrast": (
            coherent_fraction * total_variation
        ).astype(np.float64),
        "nb2_pearson_residual": pearson.astype(np.float64),
        "nb2_deviance": deviance.astype(np.float64),
        "nb2_fisher_log_total": fisher.astype(np.float64),
    }


def _loss_and_gradients(
    arrays: V7JointNB2Arrays,
    target_design: np.ndarray,
    sample_design: np.ndarray,
    operation_design: np.ndarray,
    coherent_coefficients: np.ndarray,
    accidental_coefficients: np.ndarray,
    total_rate_coefficients: np.ndarray,
    coherent_fraction_coefficients: np.ndarray,
    *,
    coherent_fraction_lower: float,
    coherent_fraction_upper: float,
    dispersion: float,
    dark_fraction: float,
    count_smoothing: float,
    coefficient_ridge: float,
    q_ridge: float,
    synthetic_prior: Optional[np.ndarray],
    synthetic_prior_weight: float,
) -> Tuple[float, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Dict[str, np.ndarray]]:
    coin = np.asarray(arrays.coin_counts, dtype=float)
    accidental_seed = (
        np.asarray(arrays.signal_counts, dtype=float) + count_smoothing
    ) * (
        np.asarray(arrays.idler_counts, dtype=float) + count_smoothing
    )
    accidental_logits = np.log(np.maximum(accidental_seed, 1e-12))
    accidental_logits += operation_design @ accidental_coefficients
    accidental_probability = _softmax(accidental_logits)
    coherent_delta = (target_design @ coherent_coefficients) @ operation_design.T
    coherent_delta -= np.mean(coherent_delta, axis=1, keepdims=True)
    coherent_probability = _softmax(accidental_logits + np.clip(coherent_delta, -8.0, 8.0))
    q_unit = _sigmoid(sample_design @ coherent_fraction_coefficients)
    q_range = coherent_fraction_upper - coherent_fraction_lower
    coherent_fraction = coherent_fraction_lower + q_range * q_unit
    dark_probability = np.full_like(accidental_probability, 1.0 / accidental_probability.shape[1])
    active_probability = (
        coherent_fraction[:, None] * coherent_probability
        + (1.0 - coherent_fraction[:, None]) * accidental_probability
    )
    probability = (1.0 - dark_fraction) * active_probability + dark_fraction * dark_probability
    probability = np.maximum(probability, 1e-12)
    probability /= np.sum(probability, axis=1, keepdims=True)
    total_mean = np.exp(np.clip(sample_design @ total_rate_coefficients, -5.0, 20.0))
    mean = np.maximum(total_mean[:, None] * probability, 1e-10)
    alpha = float(max(dispersion, 0.0))
    if alpha <= 1e-10:
        loss_values = mean - coin * np.log(mean)
        gradient_log_mean = mean - coin
    else:
        size = 1.0 / alpha
        loss_values = size * np.log(size + mean) + coin * (
            np.log(size + mean) - np.log(mean)
        )
        gradient_log_mean = (mean - coin) / (1.0 + alpha * mean)
    sample_weight = np.asarray(arrays.sample_weights, dtype=float).reshape(-1, 1)
    normalizer = max(float(np.sum(sample_weight)) * coin.shape[1], 1.0)
    loss = float(np.sum(sample_weight * loss_values) / normalizer)
    gradient_log_mean = sample_weight * gradient_log_mean / normalizer
    gradient_probability = gradient_log_mean / probability

    gradient_coherent_probability = (
        (1.0 - dark_fraction) * coherent_fraction[:, None] * gradient_probability
    )
    gradient_accidental_probability = (
        (1.0 - dark_fraction) * (1.0 - coherent_fraction[:, None]) * gradient_probability
    )
    gradient_coherent_logits = coherent_probability * (
        gradient_coherent_probability
        - np.sum(gradient_coherent_probability * coherent_probability, axis=1, keepdims=True)
    )
    gradient_accidental_logits = accidental_probability * (
        gradient_accidental_probability
        - np.sum(gradient_accidental_probability * accidental_probability, axis=1, keepdims=True)
    )
    gradient_base_logits = gradient_coherent_logits + gradient_accidental_logits
    gradient_coherent = target_design.T @ (gradient_coherent_logits @ operation_design)
    gradient_accidental = operation_design.T @ np.sum(gradient_base_logits, axis=0)
    gradient_total = sample_design.T @ np.sum(gradient_log_mean, axis=1)
    gradient_q = (1.0 - dark_fraction) * np.sum(
        gradient_probability * (coherent_probability - accidental_probability), axis=1,
    )
    gradient_q *= q_range * q_unit * (1.0 - q_unit)
    gradient_q_coefficients = sample_design.T @ gradient_q

    loss += 0.5 * coefficient_ridge * (
        float(np.sum(np.square(coherent_coefficients)))
        + float(np.sum(np.square(accidental_coefficients)))
    )
    loss += 0.5 * q_ridge * float(np.sum(np.square(coherent_fraction_coefficients[1:])))
    gradient_coherent += coefficient_ridge * coherent_coefficients
    gradient_accidental += coefficient_ridge * accidental_coefficients
    gradient_q_coefficients[1:] += q_ridge * coherent_fraction_coefficients[1:]
    if synthetic_prior is not None and synthetic_prior_weight > 0.0:
        prior = np.asarray(synthetic_prior, dtype=float)
        if prior.shape != coherent_coefficients.shape:
            raise ValueError("synthetic coherent prior has the wrong shape")
        difference = coherent_coefficients - prior
        loss += 0.5 * synthetic_prior_weight * float(np.sum(np.square(difference)))
        gradient_coherent += synthetic_prior_weight * difference
    return loss, (
        gradient_coherent,
        gradient_accidental,
        gradient_total,
        gradient_q_coefficients,
    ), {
        "probability": probability,
        "expected_counts": mean,
        "coherent_fraction": coherent_fraction,
        "coherent_delta": coherent_delta,
    }


def fit_v7_joint_nb2(
    system_type: str,
    arrays: V7JointNB2Arrays,
    train_mask: Any,
    geometry: OperationFrequencyGeometryV1,
    retrieval_rate_vectors: Any,
    target_feature_names: Sequence[str],
    *,
    graph_mode_count: int,
    graph_length_mhz: float,
    coherent_fraction_lower: float,
    coherent_fraction_upper: float,
    coherent_fraction_initial: float,
    nb2_dispersion: float,
    dark_fraction: float,
    dark_supervision_available: bool,
    count_smoothing: float,
    optimizer_steps: int,
    batch_size: int,
    learning_rate: float,
    coefficient_ridge: float,
    q_ridge: float,
    observed_q_prior_strength: float,
    q_refinement_grid_size: int,
    seed: int,
    coherent_prior: Optional[Any] = None,
    synthetic_prior_weight: float = 0.0,
    target_normalization: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    sample_normalization: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[V7JointNB2Model, Dict[str, Any]]:
    arrays.validate(geometry.operation_count)
    train = np.asarray(train_mask, dtype=bool).reshape(-1)
    if train.shape != (len(arrays.coin_counts),) or not np.any(train):
        raise ValueError("train_mask must select samples")
    if not 0.0 <= coherent_fraction_lower < coherent_fraction_upper <= 1.0:
        raise ValueError("coherent fraction bounds must satisfy 0 <= lower < upper <= 1")
    if not coherent_fraction_lower <= coherent_fraction_initial <= coherent_fraction_upper:
        raise ValueError("coherent_fraction_initial must lie inside the configured bounds")
    if dark_fraction > 0.0 and not dark_supervision_available:
        raise ValueError("non-zero dark_fraction requires external dark supervision")
    operation_design, operation_names, operation_audit = build_joint_operation_design(
        geometry,
        retrieval_rate_vectors,
        graph_mode_count=graph_mode_count,
        graph_length_mhz=graph_length_mhz,
    )
    target_values = np.asarray(arrays.target_features, dtype=float)
    if target_normalization is None:
        target_mean = np.mean(target_values[train], axis=0)
        target_scale = np.std(target_values[train], axis=0)
    else:
        target_mean, target_scale = (
            np.asarray(target_normalization[0], dtype=float),
            np.asarray(target_normalization[1], dtype=float),
        )
    target_scale = np.asarray(target_scale, dtype=float).copy()
    target_scale[target_scale < 1e-9] = 1.0
    target_design = np.column_stack((
        np.ones(target_values.shape[0]),
        (target_values - target_mean) / target_scale,
    ))
    sample_values = sample_context_features(arrays)
    if sample_normalization is None:
        sample_mean = np.mean(sample_values[train], axis=0)
        sample_scale = np.std(sample_values[train], axis=0)
    else:
        sample_mean, sample_scale = (
            np.asarray(sample_normalization[0], dtype=float),
            np.asarray(sample_normalization[1], dtype=float),
        )
    sample_scale = np.asarray(sample_scale, dtype=float).copy()
    sample_scale[sample_scale < 1e-9] = 1.0
    sample_design = np.column_stack((
        np.ones(sample_values.shape[0]),
        (sample_values - sample_mean) / sample_scale,
    ))

    train_indices = np.flatnonzero(train)
    target_dimension = target_design.shape[1]
    operation_dimension = operation_design.shape[1]
    coherent_coefficients = np.zeros((target_dimension, operation_dimension), dtype=np.float64)
    if coherent_prior is not None and float(synthetic_prior_weight) > 0.0:
        coherent_coefficients[:] = np.asarray(coherent_prior, dtype=float)
    accidental_coefficients = np.zeros(operation_dimension, dtype=np.float64)
    total_target = np.log(np.maximum(np.sum(arrays.coin_counts, axis=1), 1.0))
    total_rate_coefficients = np.linalg.pinv(
        sample_design[train].T @ sample_design[train] + 1e-3 * np.eye(sample_design.shape[1])
    ) @ sample_design[train].T @ total_target[train]
    coherent_fraction_coefficients = np.zeros(sample_design.shape[1], dtype=np.float64)
    initial_unit = (
        coherent_fraction_initial - coherent_fraction_lower
    ) / max(coherent_fraction_upper - coherent_fraction_lower, 1e-12)
    coherent_fraction_coefficients[0] = np.log(
        max(initial_unit, 1e-6) / max(1.0 - initial_unit, 1e-6)
    )

    parameters = [
        coherent_coefficients,
        accidental_coefficients,
        total_rate_coefficients,
        coherent_fraction_coefficients,
    ]
    first_moment = [np.zeros_like(value) for value in parameters]
    second_moment = [np.zeros_like(value) for value in parameters]
    rng = np.random.default_rng(int(seed))
    loss_history = []
    effective_batch_size = min(max(1, int(batch_size)), train_indices.size)
    for step in range(max(1, int(optimizer_steps))):
        if effective_batch_size == train_indices.size:
            batch_indices = train_indices
        else:
            batch_indices = rng.choice(train_indices, size=effective_batch_size, replace=False)
        batch = V7JointNB2Arrays(
            coin_counts=np.asarray(arrays.coin_counts)[batch_indices],
            signal_counts=np.asarray(arrays.signal_counts)[batch_indices],
            idler_counts=np.asarray(arrays.idler_counts)[batch_indices],
            target_features=np.asarray(arrays.target_features)[batch_indices],
            multi_click_counts=np.asarray(arrays.multi_click_counts)[batch_indices],
            exposure_proxy=np.asarray(arrays.exposure_proxy)[batch_indices],
            sample_weights=np.asarray(arrays.sample_weights)[batch_indices],
        )
        loss, gradients, _ = _loss_and_gradients(
            batch,
            target_design[batch_indices],
            sample_design[batch_indices],
            operation_design,
            *parameters,
            coherent_fraction_lower=coherent_fraction_lower,
            coherent_fraction_upper=coherent_fraction_upper,
            dispersion=nb2_dispersion,
            dark_fraction=dark_fraction,
            count_smoothing=count_smoothing,
            coefficient_ridge=coefficient_ridge,
            q_ridge=q_ridge,
            synthetic_prior=None if coherent_prior is None else np.asarray(coherent_prior, dtype=float),
            synthetic_prior_weight=synthetic_prior_weight,
        )
        loss_history.append(float(loss))
        time_index = step + 1
        for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
            first_moment[index] = 0.9 * first_moment[index] + 0.1 * gradient
            second_moment[index] = 0.999 * second_moment[index] + 0.001 * np.square(gradient)
            corrected_first = first_moment[index] / (1.0 - 0.9 ** time_index)
            corrected_second = second_moment[index] / (1.0 - 0.999 ** time_index)
            parameter -= float(learning_rate) * corrected_first / (
                np.sqrt(corrected_second) + 1e-8
            )
    model = V7JointNB2Model(
        system_type=system_type,
        geometry_hash=geometry.geometry_hash,
        target_feature_names=tuple(str(name) for name in target_feature_names),
        operation_feature_names=operation_names,
        operation_design=operation_design,
        target_feature_mean=target_mean,
        target_feature_scale=target_scale,
        sample_feature_mean=sample_mean,
        sample_feature_scale=sample_scale,
        coherent_coefficients=coherent_coefficients,
        accidental_coefficients=accidental_coefficients,
        total_rate_coefficients=total_rate_coefficients,
        coherent_fraction_coefficients=coherent_fraction_coefficients,
        coherent_fraction_lower=float(coherent_fraction_lower),
        coherent_fraction_upper=float(coherent_fraction_upper),
        nb2_dispersion=float(nb2_dispersion),
        dark_fraction=float(dark_fraction),
        dark_supervision_available=bool(dark_supervision_available),
        count_smoothing=float(count_smoothing),
        observed_q_prior_strength=float(observed_q_prior_strength),
        q_refinement_grid_size=int(q_refinement_grid_size),
        training_sample_count=int(np.sum(train)),
        synthetic_pretrained=coherent_prior is not None and float(synthetic_prior_weight) > 0.0,
        synthetic_prior_weight=float(synthetic_prior_weight),
    )
    train_arrays = V7JointNB2Arrays(
        coin_counts=np.asarray(arrays.coin_counts)[train],
        signal_counts=np.asarray(arrays.signal_counts)[train],
        idler_counts=np.asarray(arrays.idler_counts)[train],
        target_features=np.asarray(arrays.target_features)[train],
        multi_click_counts=np.asarray(arrays.multi_click_counts)[train],
        exposure_proxy=np.asarray(arrays.exposure_proxy)[train],
        sample_weights=np.asarray(arrays.sample_weights)[train],
    )
    train_components = model.components(train_arrays)
    report = {
        "schema": "v7_joint_nb2_fit_report_v1",
        "training_sample_count": int(np.sum(train)),
        "optimizer_steps": int(optimizer_steps),
        "initial_loss": float(loss_history[0]),
        "final_loss": float(loss_history[-1]),
        "loss_decreased": bool(loss_history[-1] < loss_history[0]),
        "train_nb2_nll_per_bin": float(nb2_nll(
            train_arrays.coin_counts,
            train_components["expected_counts"],
            nb2_dispersion,
        )),
        "train_conditional_nll_per_click": float(np.mean(_conditional_nll_per_click(
            train_arrays.coin_counts,
            train_components["probability"],
        ))),
        "train_coherent_fraction_quantiles": np.quantile(
            train_components["coherent_fraction"], (0.10, 0.50, 0.90),
        ).tolist(),
        "coherent_coefficient_norm": float(np.linalg.norm(coherent_coefficients)),
        "accidental_coefficient_norm": float(np.linalg.norm(accidental_coefficients)),
        "operation_design_audit": operation_audit,
        "global_multi_click_scope_preserved": True,
        "dark_component_fitted": bool(dark_supervision_available),
    }
    return model, report


def evaluate_joint_hidden_nll(
    model: V7JointNB2Model,
    arrays: V7JointNB2Arrays,
    sample_mask: Any,
    operation_masks: Mapping[str, np.ndarray],
    *,
    refine_coherent_fraction: bool,
) -> Tuple[Dict[str, Any], np.ndarray]:
    selected = np.asarray(sample_mask, dtype=bool)
    subset = V7JointNB2Arrays(
        coin_counts=np.asarray(arrays.coin_counts)[selected],
        signal_counts=np.asarray(arrays.signal_counts)[selected],
        idler_counts=np.asarray(arrays.idler_counts)[selected],
        target_features=np.asarray(arrays.target_features)[selected],
        multi_click_counts=np.asarray(arrays.multi_click_counts)[selected],
        exposure_proxy=np.asarray(arrays.exposure_proxy)[selected],
        sample_weights=np.asarray(arrays.sample_weights)[selected],
    )
    rows = []
    q_shift = []
    for observed_mask in operation_masks.values():
        probability, diagnostics = model.predict_hidden_probability(
            subset,
            subset.coin_counts[:, observed_mask],
            observed_mask,
            refine_coherent_fraction=refine_coherent_fraction,
        )
        rows.append(_conditional_nll_per_click(
            subset.coin_counts[:, ~observed_mask], probability,
        ))
        q_shift.append(
            diagnostics["selected_coherent_fraction"]
            - diagnostics["prior_coherent_fraction"]
        )
    by_sample = np.mean(np.stack(rows), axis=0)
    components = model.components(subset)
    return {
        "available": bool(np.any(selected)),
        "sample_count": int(np.sum(selected)),
        "hidden_conditional_nll_per_click": float(np.mean(by_sample)),
        "full_nb2_nll_per_bin": float(nb2_nll(
            subset.coin_counts,
            components["expected_counts"],
            model.nb2_dispersion,
        )),
        "coherent_fraction_quantiles": np.quantile(
            components["coherent_fraction"], (0.10, 0.50, 0.90),
        ).tolist(),
        "observed_q_refinement_abs_mean": float(np.mean(np.abs(q_shift))),
        "refine_coherent_fraction": bool(refine_coherent_fraction),
    }, by_sample


__all__ = [
    "JOINT_NB2_SCHEMA",
    "JOINT_PHYSICS_FACTOR_NAMES",
    "SAMPLE_CONTEXT_NAMES",
    "V7JointNB2Arrays",
    "V7JointNB2Model",
    "build_joint_operation_design",
    "evaluate_joint_hidden_nll",
    "fit_v7_joint_nb2",
    "joint_physics_factors",
    "sample_context_features",
]
