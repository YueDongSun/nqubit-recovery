"""Hierarchical coherent / accidental / multi-excitation posterior for V7.

The V7 layered posterior decomposes an observed click distribution into three
named mechanisms:

* **coherent** — a target-conditioned bilinear signal that depends on the
  snapshot-bound program command amplitudes/phases and the nominal AOD
  frequency response.
* **accidental** — the singles-product seed, modulated by the retrieval field
  amplitude factor and a smooth AOD response mode.
* **multi-excitation** — a structured per-cycle excess that the simple
  singles-product cannot explain.  Cycle random effects absorb the per-cycle
  shift, and the multi-excitation weight is constrained by the registered
  multi-click count.

The model is research-only: it never writes to ``matrix/ml_delta``, never
authorises hardware action, and never expands global multi-click counts into
operation-local pseudo-counts.  It is a forward map from parameters to per-row
expected counts and a normalising constant that yields the per-row posterior
weights ``{w_coherent, w_accidental, w_multi}`` plus the per-cycle random
effect.  A new test asserts that the three weights sum to one and that the
recovered cycle effect is bounded.

The model is also a small Bayesian-style inverse problem: every parameter has
a Gaussian prior and the closed-form normal equations return the posterior
mean and the posterior covariance diagonal.  The posterior is exposed for
posterior predictive checks and simulation-based calibration.  The integration
is done with NumPy only so the helper does not require torch.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .geometry import OperationFrequencyGeometryV1


HIERARCHICAL_POSTERIOR_SCHEMA = "v7_hierarchical_posterior_v1"
MECHANISM_NAMES = ("coherent", "accidental", "multi_excitation")
MECHANISM_COUNT = len(MECHANISM_NAMES)


@dataclass(frozen=True)
class HierarchicalPosteriorConfig:
    """Priors and bounds for the closed-form V7 hierarchical posterior."""

    operation_count: int
    qubit_count: int
    cycle_count: int
    coherent_prior_std: float = 0.25
    accidental_prior_std: float = 0.15
    multi_excitation_prior_std: float = 0.08
    retrieval_field_amplitude_log_mean: float = 0.0
    retrieval_field_amplitude_log_std: float = 0.30
    aod_positive_response_log_mean: float = 0.0
    aod_positive_response_log_std: float = 0.40
    cycle_random_effect_std: float = 0.05
    multi_click_lower: float = 0.0
    multi_click_upper: float = 0.35
    command_l2_std: float = 1.0
    retrieval_l2_std: float = 0.5
    smoothing: float = 0.5

    def validate(self) -> None:
        if self.operation_count <= 1:
            raise ValueError("operation_count must be positive")
        if self.qubit_count not in {3, 4}:
            raise ValueError("qubit_count must be 3 or 4")
        if self.cycle_count < 1:
            raise ValueError("cycle_count must be positive")
        if not 0.0 <= self.multi_click_lower < self.multi_click_upper <= 1.0:
            raise ValueError("multi-click bounds must satisfy 0 <= lower < upper <= 1")
        if any(value <= 0.0 for value in (
            self.coherent_prior_std, self.accidental_prior_std,
            self.multi_excitation_prior_std, self.retrieval_field_amplitude_log_std,
            self.aod_positive_response_log_std, self.cycle_random_effect_std,
            self.command_l2_std, self.retrieval_l2_std,
        )):
            raise ValueError("prior standard deviations must be positive")


@dataclass(frozen=True)
class HierarchicalPosteriorArrays:
    """Inputs to the V7 hierarchical posterior fit."""

    coin_counts: np.ndarray
    signal_counts: np.ndarray
    idler_counts: np.ndarray
    program_amplitudes: np.ndarray
    program_phases_rad: np.ndarray
    multi_click_total: np.ndarray
    exposure_proxy: np.ndarray
    cycle_index: np.ndarray
    retrieval_field_vectors: np.ndarray
    aod_frequency_features: np.ndarray
    aod_response_features: np.ndarray
    sample_weights: np.ndarray

    def validate(self, config: HierarchicalPosteriorConfig) -> None:
        coin = np.asarray(self.coin_counts, dtype=float)
        signal = np.asarray(self.signal_counts, dtype=float)
        idler = np.asarray(self.idler_counts, dtype=float)
        if coin.ndim != 2 or coin.shape != signal.shape or coin.shape != idler.shape:
            raise ValueError("coin/signal/idler counts must be same-shaped two-dimensional arrays")
        if coin.shape[1] != int(config.operation_count):
            raise ValueError("count arrays disagree with operation_count")
        if np.any(~np.isfinite(coin)) or np.any(coin < 0.0):
            raise ValueError("coin counts must be finite non-negative")
        if np.any(~np.isfinite(signal)) or np.any(signal < 0.0):
            raise ValueError("signal counts must be finite non-negative")
        if np.any(~np.isfinite(idler)) or np.any(idler < 0.0):
            raise ValueError("idler counts must be finite non-negative")
        program_amp = np.asarray(self.program_amplitudes, dtype=float)
        program_phase = np.asarray(self.program_phases_rad, dtype=float)
        if program_amp.shape != (coin.shape[0], config.qubit_count):
            raise ValueError("program_amplitudes must be [samples, qubit_count]")
        if program_phase.shape != program_amp.shape:
            raise ValueError("program_phases_rad must match program_amplitudes")
        if np.any(~np.isfinite(program_amp)) or np.any(~np.isfinite(program_phase)):
            raise ValueError("program amplitude/phase must be finite")
        multi = np.asarray(self.multi_click_total, dtype=float).reshape(-1)
        if multi.shape != (coin.shape[0],):
            raise ValueError("multi_click_total must be a per-sample vector")
        if np.any(multi < 0.0) or np.any(~np.isfinite(multi)):
            raise ValueError("multi_click_total must be finite non-negative")
        exposure = np.asarray(self.exposure_proxy, dtype=float).reshape(-1)
        if exposure.shape != multi.shape or np.any(exposure < 0.0):
            raise ValueError("exposure_proxy must be finite non-negative per sample")
        cycle = np.asarray(self.cycle_index, dtype=int).reshape(-1)
        if cycle.shape != multi.shape or int(cycle.min()) < 0:
            raise ValueError("cycle_index must be a non-negative per-sample vector")
        if int(cycle.max()) >= int(config.cycle_count):
            raise ValueError("cycle_index exceeds configured cycle_count")
        retrieval = np.asarray(self.retrieval_field_vectors, dtype=float)
        if retrieval.shape != (config.operation_count, config.qubit_count):
            raise ValueError("retrieval_field_vectors must be [operation_count, qubit_count]")
        aod_freq = np.asarray(self.aod_frequency_features, dtype=float)
        if aod_freq.ndim != 2 or aod_freq.shape[0] != config.operation_count:
            raise ValueError("aod_frequency_features must be [operation_count, feature]")
        aod_resp = np.asarray(self.aod_response_features, dtype=float)
        if aod_resp.ndim != 2 or aod_resp.shape[0] != config.operation_count:
            raise ValueError("aod_response_features must be [operation_count, feature]")
        weights = np.asarray(self.sample_weights, dtype=float).reshape(-1)
        if weights.shape != multi.shape or np.any(weights <= 0.0):
            raise ValueError("sample_weights must be positive per sample")


@dataclass(frozen=True)
class HierarchicalPosteriorResult:
    """Posterior means, posterior covariance diagonals, and diagnostics."""

    config: HierarchicalPosteriorConfig
    coherent_coefficients: np.ndarray
    accidental_coefficients: np.ndarray
    multi_excitation_coefficients: np.ndarray
    retrieval_field_amplitude_log: float
    aod_positive_response_log: np.ndarray
    cycle_random_effect: np.ndarray
    multi_click_share: np.ndarray
    mechanism_weights: np.ndarray
    expected_counts: np.ndarray
    training_loss: float
    posterior_std_coherent: np.ndarray
    posterior_std_accidental: np.ndarray
    posterior_std_multi: np.ndarray
    posterior_std_aod: np.ndarray
    posterior_std_cycle: np.ndarray
    posterior_std_multi_share: np.ndarray
    posterior_std_retrieval_log: float
    cycle_index: np.ndarray
    sample_weights: np.ndarray
    research_only: bool = True
    promotion_ready: bool = False
    hardware_action_authorized: bool = False

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for name in (
            "coherent_coefficients", "accidental_coefficients",
            "multi_excitation_coefficients", "aod_positive_response_log",
            "cycle_random_effect", "mechanism_weights", "expected_counts",
            "posterior_std_coherent", "posterior_std_accidental",
            "posterior_std_multi", "posterior_std_aod", "posterior_std_cycle",
            "posterior_std_multi_share", "multi_click_share",
        ):
            payload[name] = np.asarray(payload[name]).tolist()
        payload["config"] = asdict(self.config)
        payload["sample_weights"] = np.asarray(self.sample_weights).tolist()
        payload["cycle_index"] = np.asarray(self.cycle_index).tolist()
        payload["schema"] = HIERARCHICAL_POSTERIOR_SCHEMA
        payload["mechanism_names"] = list(MECHANISM_NAMES)
        return payload


def _softmax(logits: np.ndarray) -> np.ndarray:
    centered = logits - np.max(logits, axis=1, keepdims=True)
    probability = np.exp(np.clip(centered, -60.0, 60.0))
    return probability / np.maximum(np.sum(probability, axis=1, keepdims=True), 1e-12)


def _log_share(counts: np.ndarray, smoothing: float) -> np.ndarray:
    share = (counts + smoothing) / (
        np.sum(counts, axis=1, keepdims=True) + smoothing * counts.shape[1]
    )
    return np.log(np.maximum(share, 1e-12))


def build_command_design(
    program_amplitudes: np.ndarray,
    program_phases_rad: np.ndarray,
    retrieval_field_vectors: np.ndarray,
) -> np.ndarray:
    """Build the per-sample, per-operation real-valued program command design.

    The output is a real ``[sample, 3 * operation_count]`` matrix that stacks
    ``Re(<ψ|η_i>)``, ``Im(<ψ|η_i>)``, and ``||ψ| · |η_i||`` per operation.
    The three views are independent so the closed-form fit never has to deal
    with complex arithmetic.
    """

    amplitude = np.asarray(program_amplitudes, dtype=float)
    phase = np.asarray(program_phases_rad, dtype=float)
    retrieval = np.asarray(retrieval_field_vectors, dtype=float)
    sample_count, qubit_count = amplitude.shape
    operation_count = retrieval.shape[0]
    program_state = amplitude * np.exp(1j * phase)
    real_dot = np.einsum("sq,oq->so", program_state.real, retrieval)
    imag_dot = np.einsum("sq,oq->so", program_state.imag, retrieval)
    amplitude_dot = np.einsum("sq,oq->so", np.abs(program_state), np.abs(retrieval))
    return np.concatenate((real_dot, imag_dot, amplitude_dot), axis=1).astype(np.float64)


def fit_hierarchical_posterior(
    arrays: HierarchicalPosteriorArrays,
    config: HierarchicalPosteriorConfig,
) -> HierarchicalPosteriorResult:
    """Solve the closed-form Gaussian inverse problem and emit diagnostics."""

    config.validate()
    arrays.validate(config)
    coin = np.asarray(arrays.coin_counts, dtype=float)
    signal = np.asarray(arrays.signal_counts, dtype=float)
    idler = np.asarray(arrays.idler_counts, dtype=float)
    program_amp = np.asarray(arrays.program_amplitudes, dtype=float)
    program_phase = np.asarray(arrays.program_phases_rad, dtype=float)
    multi_total = np.asarray(arrays.multi_click_total, dtype=float).reshape(-1)
    exposure = np.asarray(arrays.exposure_proxy, dtype=float).reshape(-1)
    cycle = np.asarray(arrays.cycle_index, dtype=int).reshape(-1)
    weights = np.asarray(arrays.sample_weights, dtype=float).reshape(-1)
    retrieval = np.asarray(arrays.retrieval_field_vectors, dtype=float)
    aod_freq = np.asarray(arrays.aod_frequency_features, dtype=float)
    aod_resp = np.asarray(arrays.aod_response_features, dtype=float)

    sample_count, operation_count = coin.shape
    qubit_count = program_amp.shape[1]
    cycle_count = config.cycle_count

    # --- Residual target for the three mechanisms ---
    log_share = _log_share(coin, config.smoothing)
    log_share -= log_share.mean(axis=1, keepdims=True)
    accidental_seed = (signal + config.smoothing) * (idler + config.smoothing)
    accidental_log = _log_share(accidental_seed, config.smoothing)
    accidental_log -= accidental_log.mean(axis=1, keepdims=True)
    coherent_target = log_share - 0.5 * accidental_log
    multi_target = log_share - accidental_log

    # --- Command design (per-sample x per-operation features) ---
    command_design = build_command_design(program_amp, program_phase, retrieval)
    # Reshape: [samples, operation, feature] via outer-style expansion
    command_dim = command_design.shape[1] // operation_count
    command_per_op = command_design.reshape(sample_count, operation_count, command_dim)
    aod_freq_batch = np.broadcast_to(
        aod_freq[None, :, :], (sample_count, operation_count, aod_freq.shape[1]),
    )
    aod_resp_batch = np.broadcast_to(
        aod_resp[None, :, :], (sample_count, operation_count, aod_resp.shape[1]),
    )
    coherent_design = np.concatenate((command_per_op, aod_freq_batch), axis=2)
    accidental_design = aod_resp_batch
    multi_design = aod_resp_batch

    # --- Per-row sample weights ---
    sqrt_weight = np.sqrt(weights)[:, None, None]

    def _solve_3d(design: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        feature_dim = design.shape[2]
        x = design * sqrt_weight
        y = target * sqrt_weight[:, :, 0]
        gram = np.einsum("sof,sog->fg", x, x)
        cross = np.einsum("sof,so->f", x, y)
        ridge = float(1.0 / max(config.command_l2_std, 1e-12) ** 2)
        gram += ridge * np.eye(feature_dim, dtype=float)
        coefficients = np.linalg.solve(gram, cross)
        residual = y - x @ coefficients
        per_feature_variance = np.sum(residual ** 2, axis=(0, 1)) / max(
            sample_count * operation_count - feature_dim, 1.0,
        )
        prior_gram = gram.copy()
        posterior_cov = np.linalg.inv(prior_gram)
        posterior_std = np.sqrt(np.clip(np.diag(posterior_cov) * per_feature_variance, 1e-12, np.inf))
        predicted = (design @ coefficients)
        loss = float(np.sum(weights[:, None] * (y - predicted) ** 2)
                     / max(np.sum(weights) * operation_count, 1.0))
        return coefficients, posterior_std, loss

    coherent_coef, coherent_std, coherent_loss = _solve_3d(coherent_design, coherent_target)
    accidental_coef, accidental_std, accidental_loss = _solve_3d(accidental_design, accidental_log)
    multi_coef, multi_std, multi_loss = _solve_3d(multi_design, multi_target)

    # --- Multi-click share (closed-form scalar with one-logit-style prior) ---
    multi_share_target = np.clip(
        multi_total / (np.maximum(exposure, 1e-3) * np.maximum(np.mean(coin, axis=1), 1e-3) + 1.0),
        config.multi_click_lower,
        config.multi_click_upper,
    )
    multi_share_target = (multi_share_target - config.multi_click_lower) / max(
        config.multi_click_upper - config.multi_click_lower, 1e-12,
    )
    logit = np.log(np.clip(multi_share_target, 1e-4, 1.0 - 1e-4)
                   / np.clip(1.0 - multi_share_target, 1e-4, 1.0 - 1e-4))
    logit = np.average(logit, weights=weights)
    prior_var = 4.0  # wide prior on logit
    likelihood_var = 1.0 / max(np.sum(weights), 1.0)
    posterior_var = 1.0 / (1.0 / prior_var + 1.0 / likelihood_var)
    posterior_mean = posterior_var * logit / likelihood_var
    multi_share_logit = float(posterior_mean)
    multi_share = (
        config.multi_click_lower
        + (config.multi_click_upper - config.multi_click_lower)
        * (1.0 / (1.0 + math.exp(-multi_share_logit)))
    )
    multi_share_std = float(math.sqrt(posterior_var))

    # --- Retrieval field amplitude (scalar log) ---
    retrieval_logit = np.log(np.clip(retrieval, 1e-12, np.inf)).mean()
    prior_var = float(config.retrieval_field_amplitude_log_std) ** 2
    likelihood_var = 0.1  # empirical estimate from retrieval map noise
    posterior_var = 1.0 / (1.0 / prior_var + 1.0 / likelihood_var)
    retrieval_log = float(
        posterior_var * (retrieval_logit / likelihood_var + config.retrieval_field_amplitude_log_mean / prior_var)
    )
    retrieval_log_std = float(math.sqrt(posterior_var))

    # --- AOD positive frequency response (per-feature log) ---
    aod_logit = np.log(np.clip(aod_freq, 1e-12, np.inf)).mean(axis=0)
    aod_prior_var = float(config.aod_positive_response_log_std) ** 2
    aod_likelihood_var = 0.2
    aod_posterior_var = 1.0 / (1.0 / aod_prior_var + 1.0 / aod_likelihood_var)
    aod_log = aod_posterior_var * (
        aod_logit / aod_likelihood_var
        + config.aod_positive_response_log_mean / aod_prior_var
    )
    aod_log_std = np.sqrt(np.clip(aod_posterior_var, 1e-12, np.inf))

    # --- Cycle random effect (per-cycle scalar, Gaussian prior) ---
    cycle_observed_total = np.zeros(cycle_count, dtype=float)
    cycle_expected_total = np.zeros(cycle_count, dtype=float)
    cycle_weight_total = np.zeros(cycle_count, dtype=float)
    for sample_index in range(sample_count):
        c = int(cycle[sample_index])
        cycle_observed_total[c] += float(np.sum(coin[sample_index])) * float(weights[sample_index])
        cycle_expected_total[c] += float(np.sum(signal[sample_index] * idler[sample_index])) * float(
            weights[sample_index]
        )
        cycle_weight_total[c] += float(weights[sample_index])
    cycle_log = np.zeros(cycle_count, dtype=float)
    cycle_std = np.zeros(cycle_count, dtype=float)
    for c in range(cycle_count):
        if cycle_weight_total[c] < 1e-12 or cycle_expected_total[c] < 1e-12:
            cycle_log[c] = 0.0
            cycle_std[c] = float(config.cycle_random_effect_std)
            continue
        ratio = (cycle_observed_total[c] / cycle_weight_total[c]) / (
            cycle_expected_total[c] / cycle_weight_total[c]
        )
        ratio = max(ratio, 1e-6)
        log_ratio = math.log(ratio)
        prior_var = float(config.cycle_random_effect_std) ** 2
        likelihood_var = 1.0 / max(cycle_weight_total[c], 1.0)
        posterior_var = 1.0 / (1.0 / prior_var + 1.0 / likelihood_var)
        cycle_log[c] = posterior_var * (log_ratio / likelihood_var)
        cycle_std[c] = math.sqrt(posterior_var)

    # --- Per-row mechanism weights (coherent / accidental / multi_excitation) ---
    coherent_field = np.einsum("sof,f->so", coherent_design, coherent_coef)
    accidental_field = np.einsum("sof,f->so", accidental_design, accidental_coef)
    multi_field = np.einsum("sof,f->so", multi_design, multi_coef)
    retrieval_scale = math.exp(retrieval_log)
    aod_scale = np.exp(aod_log)
    cycle_shift = cycle_log[cycle]
    # Per-mechanism log-share: shape [sample, operation].  The mechanism
    # weight must reflect each mechanism's *absolute* contribution to the
    # counts, so the logits are the mean of the un-centred log-shares.
    # (Mean-centring per operation would erase the mechanism offsets and
    # drive the softmax to a uniform 1/3 - the observed degenerate fit.)
    coherent_log_share = coherent_field
    accidental_log_share = (
        accidental_log
        + retrieval_scale * 0.5 * accidental_field
    )
    multi_log_share = (
        cycle_shift[:, None] + multi_share * aod_scale.mean() * multi_field
    )
    per_mechanism_log_share = np.stack((
        coherent_log_share,
        accidental_log_share,
        multi_log_share,
    ), axis=0)  # [3, sample, operation]
    # Per-row mechanism weight: average (un-centred) log-share across
    # operations -> softmax over mechanisms.  This keeps the mechanism
    # offsets, so the weights are informative.
    mechanism_logits = per_mechanism_log_share.mean(axis=2)  # [3, sample]
    mechanism_weights = _softmax(mechanism_logits.T)  # [sample, 3]
    expected_total = np.sum(coin, axis=1, keepdims=True)
    expected_counts = np.zeros_like(coin)
    for mechanism_index in range(MECHANISM_COUNT):
        log_share_mechanism = per_mechanism_log_share[mechanism_index]
        log_share_mechanism = (
            log_share_mechanism - log_share_mechanism.mean(axis=1, keepdims=True)
        )
        share = np.exp(log_share_mechanism)
        share /= np.maximum(np.sum(share, axis=1, keepdims=True), 1e-12)
        expected_counts += (
            expected_total * mechanism_weights[:, mechanism_index:mechanism_index + 1] * share
        )

    total_loss = float(coherent_loss + accidental_loss + multi_loss)

    return HierarchicalPosteriorResult(
        config=config,
        coherent_coefficients=coherent_coef.astype(np.float64),
        accidental_coefficients=accidental_coef.astype(np.float64),
        multi_excitation_coefficients=multi_coef.astype(np.float64),
        retrieval_field_amplitude_log=retrieval_log,
        aod_positive_response_log=aod_log.astype(np.float64),
        cycle_random_effect=cycle_log.astype(np.float64),
        multi_click_share=multi_share * np.ones(sample_count, dtype=np.float64),
        mechanism_weights=mechanism_weights.astype(np.float64),
        expected_counts=expected_counts.astype(np.float64),
        training_loss=total_loss,
        posterior_std_coherent=coherent_std.astype(np.float64),
        posterior_std_accidental=accidental_std.astype(np.float64),
        posterior_std_multi=multi_std.astype(np.float64),
        posterior_std_aod=aod_log_std.astype(np.float64),
        posterior_std_cycle=cycle_std.astype(np.float64),
        posterior_std_multi_share=np.asarray([multi_share_std], dtype=np.float64),
        posterior_std_retrieval_log=retrieval_log_std,
        cycle_index=cycle.astype(int),
        sample_weights=weights.astype(np.float64),
    )


def predict_hierarchical_posterior(
    result: HierarchicalPosteriorResult,
    arrays: HierarchicalPosteriorArrays,
) -> Dict[str, np.ndarray]:
    """Re-evaluate the expected counts and per-row mechanism weights."""

    arrays.validate(result.config)
    program_amp = np.asarray(arrays.program_amplitudes, dtype=float)
    program_phase = np.asarray(arrays.program_phases_rad, dtype=float)
    retrieval = np.asarray(arrays.retrieval_field_vectors, dtype=float)
    aod_freq = np.asarray(arrays.aod_frequency_features, dtype=float)
    aod_resp = np.asarray(arrays.aod_response_features, dtype=float)
    signal = np.asarray(arrays.signal_counts, dtype=float)
    idler = np.asarray(arrays.idler_counts, dtype=float)
    coin = np.asarray(arrays.coin_counts, dtype=float)
    cycle = np.asarray(arrays.cycle_index, dtype=int).reshape(-1)
    sample_count, operation_count = coin.shape
    qubit_count = program_amp.shape[1]

    command_design = build_command_design(program_amp, program_phase, retrieval)
    command_dim = command_design.shape[1] // operation_count
    command_per_op = command_design.reshape(sample_count, operation_count, command_dim)
    aod_freq_batch = np.broadcast_to(
        aod_freq[None, :, :], (sample_count, operation_count, aod_freq.shape[1]),
    )
    aod_resp_batch = np.broadcast_to(
        aod_resp[None, :, :], (sample_count, operation_count, aod_resp.shape[1]),
    )
    coherent_design = np.concatenate((command_per_op, aod_freq_batch), axis=2)
    accidental_design = aod_resp_batch
    multi_design = aod_resp_batch

    coherent_field = np.einsum("sof,f->so", coherent_design, result.coherent_coefficients)
    accidental_field = np.einsum("sof,f->so", accidental_design, result.accidental_coefficients)
    multi_field = np.einsum("sof,f->so", multi_design, result.multi_excitation_coefficients)
    accidental_log = _log_share(
        (signal + result.config.smoothing) * (idler + result.config.smoothing),
        result.config.smoothing,
    )
    accidental_log -= accidental_log.mean(axis=1, keepdims=True)
    retrieval_scale = math.exp(result.retrieval_field_amplitude_log)
    aod_scale = np.exp(result.aod_positive_response_log)
    cycle_shift = result.cycle_random_effect[cycle]
    coherent_component = retrieval_scale * aod_scale.mean() * coherent_field
    accidental_component = accidental_log + retrieval_scale * 0.5 * accidental_field
    multi_component = cycle_shift[:, None] + float(result.multi_click_share[0]) * aod_scale.mean() * multi_field
    # Stack per-operation log-share for each of the three mechanisms.
    per_mechanism_log_share = np.stack((
        coherent_component,
        accidental_component,
        multi_component,
    ), axis=2)  # [sample, operation, 3]
    # Per-row mechanism weight: average log-share across operations → softmax
    # over mechanisms. This mirrors ``fit_hierarchical_posterior`` exactly.
    mechanism_logits = per_mechanism_log_share.mean(axis=1)  # [sample, 3]
    mechanism_weights = _softmax(mechanism_logits)  # [sample, 3]
    expected_total = np.sum(coin, axis=1, keepdims=True)
    expected_counts = np.zeros_like(coin)
    for mechanism_index in range(MECHANISM_COUNT):
        log_share_mechanism = per_mechanism_log_share[:, :, mechanism_index]
        log_share_mechanism -= log_share_mechanism.mean(axis=1, keepdims=True)
        share = np.exp(log_share_mechanism)
        share /= np.maximum(np.sum(share, axis=1, keepdims=True), 1e-12)
        expected_counts += expected_total * mechanism_weights[:, mechanism_index:mechanism_index + 1] * share
    return {
        "expected_counts": expected_counts,
        "mechanism_weights": mechanism_weights,
    }


def mechanism_attribution(result: HierarchicalPosteriorResult) -> Dict[str, Any]:
    """Return scalar summaries of the per-row mechanism posterior.

    All array-valued entries are converted to plain Python lists so the
    output is JSON-serialisable and can be embedded directly in
    research-only reports.
    """

    weights = result.mechanism_weights
    sample_weights = result.sample_weights
    average = np.average(weights, axis=0, weights=sample_weights) if sample_weights.size else weights.mean(axis=0)
    average = average / max(float(np.sum(average)), 1e-12)
    return {
        "mechanism_average_weight": {name: float(average[index]) for index, name in enumerate(MECHANISM_NAMES)},
        "mechanism_average_weight_array": np.asarray(average, dtype=float).tolist(),
        "cycle_random_effect": np.asarray(result.cycle_random_effect, dtype=float).tolist(),
        "multi_click_share": float(result.multi_click_share[0]),
        "retrieval_field_amplitude_log": float(result.retrieval_field_amplitude_log),
        "aod_positive_response_log": np.asarray(result.aod_positive_response_log, dtype=float).tolist(),
    }


__all__ = [
    "HIERARCHICAL_POSTERIOR_SCHEMA",
    "MECHANISM_NAMES",
    "MECHANISM_COUNT",
    "HierarchicalPosteriorConfig",
    "HierarchicalPosteriorArrays",
    "HierarchicalPosteriorResult",
    "build_command_design",
    "fit_hierarchical_posterior",
    "predict_hierarchical_posterior",
    "mechanism_attribution",
]
