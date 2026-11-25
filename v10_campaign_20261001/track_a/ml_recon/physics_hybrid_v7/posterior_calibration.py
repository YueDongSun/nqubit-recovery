"""Posterior predictive checks and simulation-based calibration for V7.

The posterior-predictive check (PPC) draws synthetic counts from the fitted
posterior and reports how often the observed target statistics fall inside the
predictive distribution.  The simulation-based calibration (SBC) reranks the
posterior mean against draws from the prior and reports the empirical CDF of
the rank; well-calibrated posteriors show a uniform CDF.

Both checks are research-only and use the same set of synthetic forward maps
that generated the training data so that the two checks probe the same
identifiability assumptions.  No hardware authority is touched and no count
subtraction is ever executed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from .hierarchical_posterior import (
    HIERARCHICAL_POSTERIOR_SCHEMA,
    HierarchicalPosteriorArrays,
    HierarchicalPosteriorResult,
    MECHANISM_NAMES,
    build_command_design,
    fit_hierarchical_posterior,
    predict_hierarchical_posterior,
)


CALIBRATION_SCHEMA = "v7_posterior_calibration_v1"
SBC_BIN_COUNT = 20


@dataclass(frozen=True)
class PosteriorPredictiveDiagnostics:
    """Aggregate PPC diagnostics across multiple draws and statistics."""

    statistic_names: Tuple[str, ...]
    empirical_coverage: np.ndarray
    median_predictive_minus_observed: np.ndarray
    predictive_intervals: Tuple[np.ndarray, ...]
    draw_count: int
    research_only: bool = True
    promotion_ready: bool = False
    hardware_action_authorized: bool = False

    def to_json_dict(self) -> Dict[str, Any]:
        payload = {
            "schema": CALIBRATION_SCHEMA,
            "statistic_names": list(self.statistic_names),
            "empirical_coverage": np.asarray(self.empirical_coverage).tolist(),
            "median_predictive_minus_observed": np.asarray(self.median_predictive_minus_observed).tolist(),
            "predictive_intervals": [
                np.asarray(interval).tolist() for interval in self.predictive_intervals
            ],
            "draw_count": int(self.draw_count),
            "research_only": True,
            "promotion_ready": False,
            "hardware_action_authorized": False,
        }
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


def _statistic_names() -> Tuple[str, ...]:
    return (
        "log_total_count",
        "max_share",
        "entropy",
        "first_moment",
        "second_central_moment",
        "mechanism_weight_coherent",
        "mechanism_weight_accidental",
        "mechanism_weight_multi",
    )


def _compute_statistics(coin: np.ndarray, mechanism_weights: np.ndarray) -> np.ndarray:
    sample_count = coin.shape[0]
    out = np.zeros((sample_count, len(_statistic_names())), dtype=float)
    share = (coin + 0.5) / (np.sum(coin, axis=1, keepdims=True) + 0.5 * coin.shape[1])
    out[:, 0] = np.log1p(np.sum(coin, axis=1))
    out[:, 1] = np.max(share, axis=1)
    out[:, 2] = -np.sum(share * np.log(np.clip(share, 1e-12, None)), axis=1)
    out[:, 3] = np.sum(share * np.arange(coin.shape[1])[None, :], axis=1)
    out[:, 4] = np.sum(share * (np.arange(coin.shape[1])[None, :] - out[:, 3:4]) ** 2, axis=1)
    out[:, 5] = mechanism_weights[:, 0]
    out[:, 6] = mechanism_weights[:, 1]
    out[:, 7] = mechanism_weights[:, 2]
    return out


def _draw_predictive_counts(
    posterior: HierarchicalPosteriorResult,
    arrays: HierarchicalPosteriorArrays,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw one synthetic count realisation and mechanism weights."""

    prediction = predict_hierarchical_posterior(posterior, arrays)
    expected_counts = prediction["expected_counts"]
    sample_count, operation_count = expected_counts.shape
    draw = np.zeros_like(expected_counts)
    for sample_index in range(sample_count):
        draw[sample_index] = rng.multinomial(
            max(int(np.sum(arrays.coin_counts[sample_index])), 1),
            expected_counts[sample_index] / max(np.sum(expected_counts[sample_index]), 1e-12),
        )
    mechanism_weights = prediction["mechanism_weights"]
    # Add mechanism-noise so the predictive weights are not deterministic.
    noise = rng.normal(0.0, 0.05, size=mechanism_weights.shape)
    noisy_logits = np.log(np.maximum(mechanism_weights, 1e-12)) + noise
    mechanism_weights = _softmax(noisy_logits)
    return draw.astype(np.float64), mechanism_weights


def posterior_predictive_check(
    posterior: HierarchicalPosteriorResult,
    arrays: HierarchicalPosteriorArrays,
    *,
    draw_count: int = 200,
    coverage_levels: Sequence[float] = (0.5, 0.9, 0.95),
    seed: int = 20260807,
) -> PosteriorPredictiveDiagnostics:
    """Return empirical coverage and predictive intervals for each statistic."""

    arrays.validate(posterior.config)
    prediction = predict_hierarchical_posterior(posterior, arrays)
    observed_statistics = _compute_statistics(arrays.coin_counts, prediction["mechanism_weights"])
    rng = np.random.default_rng(int(seed))
    predictive = np.zeros((int(draw_count),) + observed_statistics.shape, dtype=float)
    for draw_index in range(int(draw_count)):
        draw_coin, draw_weights = _draw_predictive_counts(posterior, arrays, rng)
        predictive[draw_index] = _compute_statistics(draw_coin, draw_weights)
    median_pred = np.median(predictive, axis=0)
    median_diff = median_pred - observed_statistics
    coverage = np.zeros((len(coverage_levels), observed_statistics.shape[1]), dtype=float)
    intervals = []
    for stat_index in range(observed_statistics.shape[1]):
        lower = np.quantile(predictive[:, :, stat_index], 0.025, axis=0)
        upper = np.quantile(predictive[:, :, stat_index], 0.975, axis=0)
        intervals.append((lower, upper))
        for level_index, level in enumerate(coverage_levels):
            q_low = (1.0 - level) / 2.0
            q_high = 1.0 - q_low
            low = np.quantile(predictive[:, :, stat_index], q_low, axis=0)
            high = np.quantile(predictive[:, :, stat_index], q_high, axis=0)
            coverage[level_index, stat_index] = float(np.mean(
                (observed_statistics[:, stat_index] >= low)
                & (observed_statistics[:, stat_index] <= high)
            ))
    intervals_2col = tuple(
        (np.asarray(lower), np.asarray(upper)) for lower, upper in intervals
    )
    return PosteriorPredictiveDiagnostics(
        statistic_names=_statistic_names(),
        empirical_coverage=coverage,
        median_predictive_minus_observed=median_diff,
        predictive_intervals=intervals_2col,
        draw_count=int(draw_count),
    )


def simulation_based_calibration(
    posterior_factory: Any,
    arrays_template: HierarchicalPosteriorArrays,
    *,
    draw_count: int = 200,
    bin_count: int = SBC_BIN_COUNT,
    seed: int = 20260807,
) -> Dict[str, Any]:
    """Re-rank the posterior mean against draws from the prior."""

    rng = np.random.default_rng(int(seed))
    rank_values: list[float] = []
    shape = arrays_template.coin_counts.shape
    bin_count = max(int(bin_count), 4)
    nll_quantiles = np.zeros(bin_count + 1, dtype=float)
    bin_counts = np.zeros(bin_count, dtype=int)
    for draw_index in range(int(draw_count)):
        prior_arrays = _draw_from_prior(arrays_template, rng)
        posterior = posterior_factory(prior_arrays)
        arrays_template.validate(posterior.config)
        observed_nll = _negative_log_likelihood(
            arrays_template.coin_counts, posterior.expected_counts,
        )
        draws = np.zeros(50, dtype=float)
        for sub_index in range(50):
            sampled = _draw_from_posterior(posterior, rng)
            draws[sub_index] = _negative_log_likelihood(
                sampled, posterior.expected_counts,
            )
        rank = float(np.mean(draws <= observed_nll))
        rank_values.append(rank)
        bin_index = min(int(rank * bin_count), bin_count - 1)
        bin_counts[bin_index] += 1
        nll_quantiles[bin_index] += observed_nll
    nll_quantiles[:-1] = np.where(bin_counts > 0, nll_quantiles[:-1] / np.maximum(bin_counts, 1), 0.0)
    empirical = np.cumsum(bin_counts) / max(np.sum(bin_counts), 1)
    expected = (np.arange(bin_count) + 0.5) / bin_count
    return {
        "schema": "v7_simulation_based_calibration_v1",
        "rank_values": np.asarray(rank_values, dtype=float),
        "bin_counts": bin_counts,
        "expected_uniform_cdf": expected,
        "empirical_cdf": empirical,
        "ks_statistic": float(np.max(np.abs(empirical - expected))),
        "draw_count": int(draw_count),
        "nll_quantile_per_bin": nll_quantiles[:-1],
        "research_only": True,
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }


def _draw_from_prior(
    arrays: HierarchicalPosteriorArrays, rng: np.random.Generator,
) -> HierarchicalPosteriorArrays:
    sample_count, operation_count = arrays.coin_counts.shape
    signal_total = np.sum(arrays.signal_counts, axis=1)
    idler_total = np.sum(arrays.idler_counts, axis=1)
    signal = np.zeros_like(arrays.signal_counts)
    idler = np.zeros_like(arrays.idler_counts)
    coin = np.zeros_like(arrays.coin_counts)
    for sample_index in range(sample_count):
        signal_share = rng.dirichlet(np.full(operation_count, 1.2))
        idler_share = rng.dirichlet(np.full(operation_count, 1.2))
        signal[sample_index] = rng.multinomial(max(int(signal_total[sample_index]), 1), signal_share)
        idler[sample_index] = rng.multinomial(max(int(idler_total[sample_index]), 1), idler_share)
        share = signal_share * idler_share
        share /= max(np.sum(share), 1e-12)
        total = max(int(rng.poisson(max(np.mean(arrays.coin_counts[sample_index]), 1.0))), 1)
        coin[sample_index] = rng.multinomial(total, share)
    return HierarchicalPosteriorArrays(
        coin_counts=coin,
        signal_counts=signal,
        idler_counts=idler,
        program_amplitudes=arrays.program_amplitudes,
        program_phases_rad=arrays.program_phases_rad,
        multi_click_total=arrays.multi_click_total,
        exposure_proxy=arrays.exposure_proxy,
        cycle_index=arrays.cycle_index,
        retrieval_field_vectors=arrays.retrieval_field_vectors,
        aod_frequency_features=arrays.aod_frequency_features,
        aod_response_features=arrays.aod_response_features,
        sample_weights=arrays.sample_weights,
    )


def _draw_from_posterior(
    posterior: HierarchicalPosteriorResult, rng: np.random.Generator,
) -> np.ndarray:
    sample_count, operation_count = posterior.expected_counts.shape
    draw = np.zeros_like(posterior.expected_counts)
    for sample_index in range(sample_count):
        share = posterior.expected_counts[sample_index] / max(
            np.sum(posterior.expected_counts[sample_index]), 1e-12
        )
        total = max(int(rng.poisson(max(np.sum(posterior.expected_counts[sample_index]), 1.0))), 1)
        draw[sample_index] = rng.multinomial(total, share)
    return draw


def _negative_log_likelihood(observed: np.ndarray, mean: np.ndarray) -> float:
    mean = np.maximum(mean, 1e-12)
    return float(-np.sum(observed * np.log(mean)) / max(np.sum(observed), 1.0))


def five_seed_aggregator(
    diagnostics_per_seed: Sequence[Mapping[str, Any]],
    *,
    ppc: Mapping[str, Any],
    sbc: Mapping[str, Any],
) -> Dict[str, Any]:
    """Aggregate five-seed posterior calibration outputs."""

    if len(diagnostics_per_seed) < 5:
        raise ValueError("five_seed_aggregator requires at least five seed reports")
    coverage = np.stack([
        np.asarray(item["empirical_coverage"], dtype=float)
        for item in diagnostics_per_seed
    ])
    median_diff = np.stack([
        np.asarray(item["median_predictive_minus_observed"], dtype=float)
        for item in diagnostics_per_seed
    ])
    return {
        "schema": "v7_five_seed_posterior_calibration_v1",
        "seed_count": len(diagnostics_per_seed),
        "statistic_names": list(diagnostics_per_seed[0]["statistic_names"]),
        "coverage_mean": coverage.mean(axis=0).tolist(),
        "coverage_std": coverage.std(axis=0).tolist(),
        "median_diff_mean": median_diff.mean(axis=0).tolist(),
        "median_diff_std": median_diff.std(axis=0).tolist(),
        "ppc_summary": dict(ppc),
        "sbc_summary": dict(sbc),
        "research_only": True,
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }


__all__ = [
    "CALIBRATION_SCHEMA",
    "SBC_BIN_COUNT",
    "PosteriorPredictiveDiagnostics",
    "posterior_predictive_check",
    "simulation_based_calibration",
    "five_seed_aggregator",
]
