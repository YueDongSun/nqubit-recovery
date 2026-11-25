"""V7 local nonlinear response plus frequency-coupled global feedback model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .geometry import FREQUENCY_CENTER_MHZ, FREQUENCY_SCALE_MHZ, OperationFrequencyGeometryV1


MODEL_SCHEMA = "v7_global_frequency_feedback_v1"
LOCAL_FEATURE_NAMES = (
    "signal_log_share_residual",
    "idler_log_share_residual",
    "signal_residual_squared",
    "idler_residual_squared",
    "signal_idler_interaction",
    "signal_residual_saturation",
    "idler_residual_saturation",
    "signal_x_frequency_interaction",
    "idler_x_frequency_interaction",
    "signal_y_frequency_interaction",
    "idler_y_frequency_interaction",
    "program_drive_amplitude",
    "program_drive_power",
    "program_drive_power_squared",
    "program_drive_power_saturation",
    "program_drive_imaginary",
    "program_drive_real",
    "program_drive_x_frequency_interaction",
    "program_drive_y_frequency_interaction",
    "program_total_power",
    "program_total_power_squared",
    "program_command_available",
)
MODEL_INPUT_AUDIT = {
    "current_matrix_directory_read": False,
    "historical_command_invented": False,
    "snapshot_path_numeric_input": False,
    "state_name_numeric_input": False,
    "target_hash_numeric_input": False,
    "hidden_coin_counts_used_for_inference": False,
    "signal_idler_measured_covariates_used": True,
    "snapshot_program_amplitude_used": True,
    "per_channel_matrix_command_claimed": False,
}


@dataclass(frozen=True)
class V7ResponseArrays:
    coin_counts: np.ndarray
    signal_counts: np.ndarray
    idler_counts: np.ndarray
    program_amplitudes: np.ndarray
    program_phases_rad: np.ndarray
    program_command_mask: np.ndarray

    def validate(self, operation_count: int) -> None:
        arrays = (
            np.asarray(self.coin_counts),
            np.asarray(self.signal_counts),
            np.asarray(self.idler_counts),
        )
        if any(array.ndim != 2 for array in arrays):
            raise ValueError("V7 response arrays must be two dimensional")
        if len({array.shape for array in arrays}) != 1:
            raise ValueError("coin/signal/idler arrays must have the same shape")
        if arrays[0].shape[1] != int(operation_count):
            raise ValueError("response array operation count disagrees with geometry")
        if any(np.any(~np.isfinite(array)) or np.any(array < 0.0) for array in arrays):
            raise ValueError("response arrays must contain finite non-negative counts")
        amplitudes = np.asarray(self.program_amplitudes, dtype=float)
        phases = np.asarray(self.program_phases_rad, dtype=float)
        command_mask = np.asarray(self.program_command_mask, dtype=float)
        if amplitudes.shape != phases.shape or amplitudes.shape[0] != arrays[0].shape[0]:
            raise ValueError("program amplitude/phase arrays have the wrong shape")
        if command_mask.shape != (arrays[0].shape[0],):
            raise ValueError("program_command_mask has the wrong shape")
        if np.any(~np.isfinite(amplitudes)) or np.any(amplitudes < 0.0):
            raise ValueError("program amplitudes must be finite and non-negative")
        if np.any(~np.isfinite(phases)) or np.any((command_mask < 0.0) | (command_mask > 1.0)):
            raise ValueError("program phases/mask contain invalid values")


@dataclass(frozen=True)
class V7PretrainingPriorV1:
    system_type: str
    geometry_hash: str
    local_coefficients: np.ndarray
    selected_rank: int
    selected_graph_blend: float
    synthetic_validation_gain: float
    synthetic_sample_count: int

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["local_coefficients"] = self.local_coefficients.tolist()
        payload["schema"] = "v7_synthetic_pretraining_prior_v1"
        return payload


@dataclass(frozen=True)
class V7GlobalFeedbackModel:
    system_type: str
    geometry_hash: str
    baseline_logit: np.ndarray
    signal_baseline_logit: np.ndarray
    idler_baseline_logit: np.ndarray
    local_coefficients: np.ndarray
    latent_loading_matrix: np.ndarray
    coupling_kernel: np.ndarray
    selected_local_ridge: float
    selected_rank: int
    selected_graph_blend: float
    smoothing: float
    residual_clip: float
    training_sample_count: int
    split_hash: str
    synthetic_pretrained: bool
    research_only: bool = True
    promotion_ready: bool = False
    hardware_action_authorized: bool = False

    def local_response_effect(
        self,
        signal_counts: Any,
        idler_counts: Any,
        program_amplitudes: Any,
        program_phases_rad: Any,
        program_command_mask: Any,
        geometry: OperationFrequencyGeometryV1,
    ) -> np.ndarray:
        features = local_nonlinear_features(
            signal_counts,
            idler_counts,
            program_amplitudes,
            program_phases_rad,
            program_command_mask,
            self.signal_baseline_logit,
            self.idler_baseline_logit,
            geometry,
            smoothing=self.smoothing,
        )
        effect = np.einsum("nof,of->no", features, self.local_coefficients)
        effect -= effect.mean(axis=1, keepdims=True)
        return np.clip(effect, -self.residual_clip, self.residual_clip)

    def predict_hidden_probability(
        self,
        observed_coin_counts: Any,
        observed_mask: Any,
        signal_counts: Any,
        idler_counts: Any,
        program_amplitudes: Any,
        program_phases_rad: Any,
        program_command_mask: Any,
        geometry: OperationFrequencyGeometryV1,
        *,
        include_local_response: bool = True,
        include_latent_coupling: bool = True,
    ) -> np.ndarray:
        if geometry.geometry_hash != self.geometry_hash:
            raise ValueError("V7 model and operation geometry hash disagree")
        mask = np.asarray(observed_mask, dtype=bool).reshape(-1)
        observed = np.asarray(observed_coin_counts, dtype=float)
        signal = np.asarray(signal_counts, dtype=float)
        idler = np.asarray(idler_counts, dtype=float)
        if mask.size != self.baseline_logit.size or not np.any(mask) or np.all(mask):
            raise ValueError("observed_mask must select a strict operation subset")
        if observed.ndim != 2 or observed.shape[1] != int(np.sum(mask)):
            raise ValueError("observed coin counts do not match observed_mask")
        if signal.shape != idler.shape or signal.shape != (observed.shape[0], mask.size):
            raise ValueError("signal/idler counts must cover every operation point")
        local_effect = (
            self.local_response_effect(
                signal, idler, program_amplitudes, program_phases_rad,
                program_command_mask, geometry,
            )
            if include_local_response else np.zeros_like(signal)
        )
        base_logits = self.baseline_logit + local_effect
        if not include_latent_coupling:
            return _softmax(base_logits[:, ~mask])
        observed_probability = _smoothed_share(observed, self.smoothing)
        observed_baseline = _softmax(base_logits[:, mask])
        observed_residual = np.log(observed_probability) - np.log(observed_baseline)
        observed_residual -= observed_residual.mean(axis=1, keepdims=True)
        loading_observed = self.latent_loading_matrix[:, mask]
        loading_hidden = self.latent_loading_matrix[:, ~mask]
        latent_score = observed_residual @ np.linalg.pinv(loading_observed)
        latent_residual = latent_score @ loading_hidden
        graph_weights = self.coupling_kernel[np.ix_(mask, ~mask)]
        graph_weights /= np.maximum(graph_weights.sum(axis=0, keepdims=True), 1e-12)
        graph_residual = observed_residual @ graph_weights
        blend = float(self.selected_graph_blend)
        global_residual = (1.0 - blend) * latent_residual + blend * graph_residual
        global_residual = np.clip(global_residual, -self.residual_clip, self.residual_clip)
        return _softmax(base_logits[:, ~mask] + global_residual)

    def response_modification(
        self,
        signal_counts: Any,
        idler_counts: Any,
        program_amplitudes: Any,
        program_phases_rad: Any,
        program_command_mask: Any,
        geometry: OperationFrequencyGeometryV1,
        *,
        maximum_log_weight: float = 0.7,
    ) -> np.ndarray:
        effect = self.local_response_effect(
            signal_counts, idler_counts, program_amplitudes, program_phases_rad,
            program_command_mask, geometry,
        )
        return np.exp(np.clip(-effect, -float(maximum_log_weight), float(maximum_log_weight)))

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for name in (
            "baseline_logit", "signal_baseline_logit", "idler_baseline_logit",
            "local_coefficients", "latent_loading_matrix", "coupling_kernel",
        ):
            payload[name] = np.asarray(payload[name]).tolist()
        payload["schema"] = MODEL_SCHEMA
        payload["local_feature_names"] = list(LOCAL_FEATURE_NAMES)
        payload["model_input_audit"] = dict(MODEL_INPUT_AUDIT)
        payload["feedback_output"] = (
            "bounded response-calibration weights and ranked operation diagnostics only"
        )
        payload["likelihood_scope"] = "conditional_operation_pattern_per_click"
        payload["absolute_rate_calibration_available"] = False
        return payload


def _smoothed_share(counts: Any, smoothing: float) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    return (values + float(smoothing)) / (
        values.sum(axis=1, keepdims=True) + float(smoothing) * values.shape[1]
    )


def _softmax(logits: Any) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    centered = values - np.max(values, axis=1, keepdims=True)
    probability = np.exp(centered)
    return probability / probability.sum(axis=1, keepdims=True)


def conditional_nll_per_click(counts: Any, probability: Any) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    predicted = np.maximum(np.asarray(probability, dtype=float), 1e-12)
    predicted /= predicted.sum(axis=1, keepdims=True)
    return -np.sum(values * np.log(predicted), axis=1) / np.maximum(
        values.sum(axis=1), 1.0,
    )


def local_nonlinear_features(
    signal_counts: Any,
    idler_counts: Any,
    program_amplitudes: Any,
    program_phases_rad: Any,
    program_command_mask: Any,
    signal_baseline_logit: np.ndarray,
    idler_baseline_logit: np.ndarray,
    geometry: OperationFrequencyGeometryV1,
    *,
    smoothing: float,
) -> np.ndarray:
    signal_log_share = np.log(_smoothed_share(signal_counts, smoothing))
    idler_log_share = np.log(_smoothed_share(idler_counts, smoothing))
    signal_residual = np.clip(signal_log_share - signal_baseline_logit, -6.0, 6.0)
    idler_residual = np.clip(idler_log_share - idler_baseline_logit, -6.0, 6.0)
    x_frequency = (
        geometry.x_mean_frequency_mhz - FREQUENCY_CENTER_MHZ
    ) / FREQUENCY_SCALE_MHZ
    y_frequency = (
        geometry.y_frequency_mhz - FREQUENCY_CENTER_MHZ
    ) / FREQUENCY_SCALE_MHZ
    amplitudes = np.asarray(program_amplitudes, dtype=float)
    phases = np.asarray(program_phases_rad, dtype=float)
    command_mask = np.asarray(program_command_mask, dtype=float).reshape(-1, 1)
    if amplitudes.shape != phases.shape or amplitudes.shape[1] != geometry.qubit_count:
        raise ValueError("program state arrays disagree with operation geometry")
    program_state = amplitudes * np.exp(1j * phases)
    basis_state = geometry.basis_amplitudes * np.exp(1j * geometry.basis_phases_rad)
    complex_drive = program_state @ basis_state.conj().T / np.sqrt(float(geometry.qubit_count))
    drive_amplitude = np.abs(complex_drive)
    drive_power = np.square(drive_amplitude)
    drive_phase = np.angle(complex_drive)
    total_power = np.mean(np.square(amplitudes), axis=1, keepdims=True)
    return np.stack((
        signal_residual,
        idler_residual,
        np.square(signal_residual),
        np.square(idler_residual),
        signal_residual * idler_residual,
        np.tanh(signal_residual),
        np.tanh(idler_residual),
        signal_residual * x_frequency,
        idler_residual * x_frequency,
        signal_residual * y_frequency,
        idler_residual * y_frequency,
        drive_amplitude,
        drive_power,
        np.square(drive_power),
        np.tanh(drive_power),
        drive_amplitude * np.sin(drive_phase),
        drive_amplitude * np.cos(drive_phase),
        drive_power * x_frequency,
        drive_power * y_frequency,
        np.broadcast_to(total_power, drive_power.shape),
        np.broadcast_to(np.square(total_power), drive_power.shape),
        np.broadcast_to(command_mask, drive_power.shape),
    ), axis=2)


def _fit_local_coefficients(
    features: np.ndarray,
    target_residual: np.ndarray,
    *,
    local_ridge: float,
    synthetic_prior: Optional[V7PretrainingPriorV1],
    synthetic_prior_weight: float,
) -> np.ndarray:
    feature_count = features.shape[2]
    global_gram = np.einsum("nof,nog->fg", features, features)
    global_cross = np.einsum("nof,no->f", features, target_residual)
    global_coefficient = np.linalg.solve(
        global_gram + float(local_ridge) * np.eye(feature_count), global_cross,
    )
    coefficients = np.empty((features.shape[1], feature_count), dtype=np.float64)
    for operation_index in range(features.shape[1]):
        x = features[:, operation_index, :]
        y = target_residual[:, operation_index]
        prior = global_coefficient
        if synthetic_prior is not None:
            prior = (
                global_coefficient
                + float(synthetic_prior_weight) * synthetic_prior.local_coefficients[operation_index]
            ) / (1.0 + float(synthetic_prior_weight))
        coefficients[operation_index] = np.linalg.solve(
            x.T @ x + float(local_ridge) * np.eye(feature_count),
            x.T @ y + float(local_ridge) * prior,
        )
    return coefficients


def _validation_nll(
    model: V7GlobalFeedbackModel,
    arrays: V7ResponseArrays,
    validation_mask: np.ndarray,
    geometry: OperationFrequencyGeometryV1,
    operation_masks: Mapping[str, np.ndarray],
) -> float:
    values = []
    for observed_mask in operation_masks.values():
        probability = model.predict_hidden_probability(
            arrays.coin_counts[validation_mask][:, observed_mask],
            observed_mask,
            arrays.signal_counts[validation_mask],
            arrays.idler_counts[validation_mask],
            arrays.program_amplitudes[validation_mask],
            arrays.program_phases_rad[validation_mask],
            arrays.program_command_mask[validation_mask],
            geometry,
        )
        values.append(conditional_nll_per_click(
            arrays.coin_counts[validation_mask][:, ~observed_mask], probability,
        ))
    return float(np.mean(np.stack(values)))


def fit_v7_global_feedback(
    arrays: V7ResponseArrays,
    geometry: OperationFrequencyGeometryV1,
    train_mask: Any,
    validation_mask: Any,
    *,
    local_ridges: Sequence[float],
    latent_ranks: Sequence[int],
    graph_blends: Sequence[float],
    operation_masks: Mapping[str, np.ndarray],
    coupling_kernel: np.ndarray,
    smoothing: float,
    residual_clip: float,
    split_hash: str,
    synthetic_prior: Optional[V7PretrainingPriorV1] = None,
    synthetic_prior_weight: float = 0.0,
) -> Tuple[V7GlobalFeedbackModel, Dict[str, Any]]:
    arrays.validate(geometry.operation_count)
    train = np.asarray(train_mask, dtype=bool)
    validation = np.asarray(validation_mask, dtype=bool)
    if train.shape != (arrays.coin_counts.shape[0],) or validation.shape != train.shape:
        raise ValueError("train/validation masks have the wrong shape")
    if not np.any(train) or not np.any(validation) or np.any(train & validation):
        raise ValueError("train and validation masks must be non-empty and disjoint")
    if synthetic_prior is not None and (
        synthetic_prior.geometry_hash != geometry.geometry_hash
        or synthetic_prior.system_type != geometry.system_type
    ):
        raise ValueError("synthetic prior and real operation geometry disagree")
    coin_probability = arrays.coin_counts[train].sum(axis=0) + float(smoothing)
    signal_probability = arrays.signal_counts[train].sum(axis=0) + float(smoothing)
    idler_probability = arrays.idler_counts[train].sum(axis=0) + float(smoothing)
    coin_probability /= coin_probability.sum()
    signal_probability /= signal_probability.sum()
    idler_probability /= idler_probability.sum()
    baseline_logit = np.log(coin_probability)
    signal_baseline_logit = np.log(signal_probability)
    idler_baseline_logit = np.log(idler_probability)
    features = local_nonlinear_features(
        arrays.signal_counts,
        arrays.idler_counts,
        arrays.program_amplitudes,
        arrays.program_phases_rad,
        arrays.program_command_mask,
        signal_baseline_logit,
        idler_baseline_logit,
        geometry,
        smoothing=smoothing,
    )
    target_residual = np.log(_smoothed_share(arrays.coin_counts, smoothing)) - baseline_logit
    candidate_scores: Dict[str, float] = {}
    best_model: Optional[V7GlobalFeedbackModel] = None
    best_key: Optional[Tuple[float, float, int, float]] = None
    for local_ridge_value in local_ridges:
        local_ridge = float(local_ridge_value)
        coefficients = _fit_local_coefficients(
            features[train], target_residual[train], local_ridge=local_ridge,
            synthetic_prior=synthetic_prior,
            synthetic_prior_weight=synthetic_prior_weight,
        )
        local_effect = np.einsum("nof,of->no", features, coefficients)
        local_effect -= local_effect.mean(axis=1, keepdims=True)
        local_effect = np.clip(local_effect, -float(residual_clip), float(residual_clip))
        residual = target_residual[train] - local_effect[train]
        residual -= residual.mean(axis=1, keepdims=True)
        _, singular_values, right_vectors = np.linalg.svd(residual, full_matrices=False)
        maximum_rank = min(
            right_vectors.shape[0],
            min(int(np.sum(mask)) for mask in operation_masks.values()),
        )
        for rank_value in latent_ranks:
            rank = int(rank_value)
            if rank <= 0 or rank > maximum_rank:
                continue
            for graph_blend_value in graph_blends:
                graph_blend = float(graph_blend_value)
                if not 0.0 <= graph_blend <= 1.0:
                    raise ValueError("graph_blends must lie in [0, 1]")
                model = V7GlobalFeedbackModel(
                    system_type=geometry.system_type,
                    geometry_hash=geometry.geometry_hash,
                    baseline_logit=baseline_logit,
                    signal_baseline_logit=signal_baseline_logit,
                    idler_baseline_logit=idler_baseline_logit,
                    local_coefficients=coefficients,
                    latent_loading_matrix=right_vectors[:rank],
                    coupling_kernel=np.asarray(coupling_kernel, dtype=np.float64),
                    selected_local_ridge=local_ridge,
                    selected_rank=rank,
                    selected_graph_blend=graph_blend,
                    smoothing=float(smoothing),
                    residual_clip=float(residual_clip),
                    training_sample_count=int(np.sum(train)),
                    split_hash=str(split_hash),
                    synthetic_pretrained=synthetic_prior is not None,
                )
                validation_nll = _validation_nll(
                    model, arrays, validation, geometry, operation_masks,
                )
                key_name = f"ridge={local_ridge:g},rank={rank},graph={graph_blend:g}"
                candidate_scores[key_name] = validation_nll
                selection_key = (validation_nll, local_ridge, rank, graph_blend)
                if best_key is None or selection_key < best_key:
                    best_model = model
                    best_key = selection_key
    if best_model is None:
        raise ValueError("V7 hyperparameter grid produced no valid candidate")
    return best_model, {
        "selected_local_ridge": best_model.selected_local_ridge,
        "selected_rank": best_model.selected_rank,
        "selected_graph_blend": best_model.selected_graph_blend,
        "validation_nll_by_candidate": candidate_scores,
        "synthetic_pretrained": best_model.synthetic_pretrained,
    }


def evaluate_v7_feedback(
    model: V7GlobalFeedbackModel,
    arrays: V7ResponseArrays,
    sample_mask: Any,
    geometry: OperationFrequencyGeometryV1,
    operation_masks: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    selected = np.asarray(sample_mask, dtype=bool)
    operation_only_rows = []
    local_rows = []
    v7_rows = []
    for observed_mask in operation_masks.values():
        hidden_counts = arrays.coin_counts[selected][:, ~observed_mask]
        operation_probability = _softmax(model.baseline_logit[~observed_mask][None, :])
        operation_only_rows.append(conditional_nll_per_click(
            hidden_counts,
            np.broadcast_to(operation_probability, hidden_counts.shape),
        ))
        local_probability = model.predict_hidden_probability(
            arrays.coin_counts[selected][:, observed_mask], observed_mask,
            arrays.signal_counts[selected], arrays.idler_counts[selected],
            arrays.program_amplitudes[selected], arrays.program_phases_rad[selected],
            arrays.program_command_mask[selected], geometry,
            include_latent_coupling=False,
        )
        local_rows.append(conditional_nll_per_click(hidden_counts, local_probability))
        v7_probability = model.predict_hidden_probability(
            arrays.coin_counts[selected][:, observed_mask], observed_mask,
            arrays.signal_counts[selected], arrays.idler_counts[selected],
            arrays.program_amplitudes[selected], arrays.program_phases_rad[selected],
            arrays.program_command_mask[selected], geometry,
        )
        v7_rows.append(conditional_nll_per_click(hidden_counts, v7_probability))
    return {
        "operation_only": np.mean(np.stack(operation_only_rows), axis=0),
        "local_nonlinear": np.mean(np.stack(local_rows), axis=0),
        "v7_global_feedback": np.mean(np.stack(v7_rows), axis=0),
    }


__all__ = [
    "LOCAL_FEATURE_NAMES",
    "MODEL_INPUT_AUDIT",
    "MODEL_SCHEMA",
    "V7GlobalFeedbackModel",
    "V7PretrainingPriorV1",
    "V7ResponseArrays",
    "conditional_nll_per_click",
    "evaluate_v7_feedback",
    "fit_v7_global_feedback",
    "local_nonlinear_features",
]
