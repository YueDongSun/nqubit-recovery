"""Pure NumPy/SciPy likelihood and group-statistics for Track B calibration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import betaln, gammaln


DELTA_BOUNDS_LOG = (-2.5, 2.5)
KAPPA_BOUNDS = (2.0, 20_000.0)
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 20_261_005
STAGE_NAMES = (
    "signal_given_read",
    "coin_given_signal",
    "idler_given_no_signal",
)
STAGE_PROBABILITY_KEYS = {
    "signal_given_read": "signal_probability",
    "coin_given_signal": "coin_given_signal_probability",
    "idler_given_no_signal": "idler_given_no_signal_probability",
}


@dataclass(frozen=True)
class PreparedLikelihood:
    """Validated immutable-shape arrays for one original evaluation fold."""

    successes: Mapping[str, np.ndarray]
    trials: Mapping[str, np.ndarray]
    probabilities: Mapping[str, np.ndarray]
    concentration: np.ndarray
    active_mask: np.ndarray
    active_per_row: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.active_mask.shape[0])

    @property
    def n_operations(self) -> int:
        return int(self.active_mask.shape[1])


def _numeric_array(name: str, values: object) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "iuf" or array.dtype.kind == "b":
        raise ValueError(f"{name} must contain real numeric values")
    try:
        numeric = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} cannot be represented as float64") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain only finite values")
    return numeric


def _integer_counts(name: str, values: object) -> np.ndarray:
    numeric = _numeric_array(name, values)
    if np.any(numeric < 0.0) or np.any(numeric > float(2**53)):
        raise ValueError(f"{name} must contain nonnegative exactly representable integers")
    if np.any(numeric != np.floor(numeric)):
        raise ValueError(f"{name} must contain nonnegative integers")
    return numeric


def _boolean_mask(name: str, values: object, expected_shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(values)
    if mask.dtype.kind != "b":
        raise ValueError(f"{name} must have boolean dtype")
    if mask.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {mask.shape}")
    return np.array(mask, dtype=np.bool_, copy=True)


def _broadcast_parameter(
    name: str,
    values: object,
    expected_shape: tuple[int, int],
    *,
    probability: bool = False,
) -> np.ndarray:
    raw = _numeric_array(name, values)
    n_rows, n_operations = expected_shape
    if raw.ndim == 0:
        reshaped = raw
    elif raw.shape == expected_shape:
        reshaped = raw
    elif raw.shape == (n_rows,):
        reshaped = raw[:, None]
    elif raw.shape == (n_rows, 1):
        reshaped = raw
    elif raw.shape == (n_operations,):
        reshaped = raw[None, :]
    elif raw.shape == (1, n_operations):
        reshaped = raw
    else:
        raise ValueError(
            f"{name} must be scalar, row-shaped, operation-shaped, or {expected_shape}; "
            f"got {raw.shape}"
        )
    try:
        result = np.array(np.broadcast_to(reshaped, expected_shape), dtype=np.float64, copy=True)
    except ValueError as exc:
        raise ValueError(f"{name} cannot broadcast to {expected_shape}") from exc
    if probability:
        if np.any(result <= 0.0) or np.any(result >= 1.0):
            raise ValueError(f"{name} must be strictly between zero and one")
    elif np.any(result <= 0.0):
        raise ValueError(f"{name} must be positive")
    return result


def prepare_likelihood(
    counts: object,
    prediction: Mapping[str, object],
    query_mask: object,
    operation_mask: object,
) -> PreparedLikelihood:
    """Validate a four-cell fold and derive the three nested binomial stages."""
    cell_counts = _integer_counts("counts", counts)
    if cell_counts.ndim != 3 or cell_counts.shape[-1] != 4:
        raise ValueError("counts must have shape [rows, operations, 4]")
    n_rows, n_operations, _ = cell_counts.shape
    if n_rows == 0 or n_operations == 0:
        raise ValueError("counts must contain at least one row and one operation")
    expected_shape = (n_rows, n_operations)

    query = _boolean_mask("query_mask", query_mask, expected_shape)
    operation = _boolean_mask("operation_mask", operation_mask, expected_shape)
    active = np.logical_and(query, operation)
    active_per_row = np.sum(active, axis=1, dtype=np.int64)
    if np.any(active_per_row == 0):
        raise ValueError("every row must have at least one query-and-operation mask entry")

    n00, n10, n01, n11 = (cell_counts[..., index] for index in range(4))
    successes = {
        "signal_given_read": n10 + n11,
        "coin_given_signal": n11,
        "idler_given_no_signal": n01,
    }
    trials = {
        "signal_given_read": n00 + n10 + n01 + n11,
        "coin_given_signal": n10 + n11,
        "idler_given_no_signal": n00 + n01,
    }

    required_prediction_keys = set(STAGE_PROBABILITY_KEYS.values()) | {"four_cell_concentration"}
    missing = sorted(required_prediction_keys - set(prediction))
    if missing:
        raise ValueError(f"prediction is missing required arrays: {missing}")

    probabilities = {
        stage: _broadcast_parameter(
            probability_key,
            prediction[probability_key],
            expected_shape,
            probability=True,
        )
        for stage, probability_key in STAGE_PROBABILITY_KEYS.items()
    }
    concentration = _broadcast_parameter(
        "four_cell_concentration", prediction["four_cell_concentration"], expected_shape
    )

    for array in (*successes.values(), *trials.values(), *probabilities.values(), concentration, active):
        array.setflags(write=False)
    active_per_row.setflags(write=False)
    return PreparedLikelihood(
        successes=successes,
        trials=trials,
        probabilities=probabilities,
        concentration=concentration,
        active_mask=active,
        active_per_row=active_per_row,
    )


def beta_binomial_log_pmf(
    success: object,
    trials: object,
    probability: object,
    concentration: object,
) -> np.ndarray:
    """Validated beta-binomial log-PMF with exact zero/full-success boundaries."""
    success_array = _integer_counts("success", success)
    trial_array = _integer_counts("trials", trials)
    if np.any(success_array > trial_array):
        raise ValueError("success must not exceed trials")
    probability_array = _numeric_array("probability", probability)
    concentration_array = _numeric_array("concentration", concentration)
    if np.any(probability_array <= 0.0) or np.any(probability_array >= 1.0):
        raise ValueError("probability must be strictly between zero and one")
    if np.any(concentration_array <= 0.0):
        raise ValueError("concentration must be positive")
    try:
        success_array, trial_array, probability_array, concentration_array = np.broadcast_arrays(
            success_array, trial_array, probability_array, concentration_array
        )
    except ValueError as exc:
        raise ValueError("success, trials, probability, and concentration shapes do not broadcast") from exc
    return _beta_binomial_log_pmf_validated(
        success_array, trial_array, probability_array, concentration_array
    )


def _beta_binomial_log_pmf_validated(
    success: np.ndarray,
    trials: np.ndarray,
    probability: np.ndarray,
    concentration: np.ndarray,
) -> np.ndarray:
    alpha = probability * concentration
    beta = (1.0 - probability) * concentration
    if np.any(alpha <= 0.0) or np.any(beta <= 0.0):
        raise ValueError("probability and concentration produce invalid beta parameters")
    return (
        gammaln(trials + 1.0)
        - gammaln(success + 1.0)
        - gammaln(trials - success + 1.0)
        + betaln(success + alpha, trials - success + beta)
        - betaln(alpha, beta)
    )


def adjusted_concentration(concentration: np.ndarray, delta_log_kappa: float) -> np.ndarray:
    """Apply a bounded log-concentration shift without changing probabilities."""
    delta = float(delta_log_kappa)
    if not np.isfinite(delta):
        raise ValueError("delta_log_kappa must be finite")
    lower_log, upper_log = np.log(KAPPA_BOUNDS)
    return np.exp(np.clip(np.log(concentration) + delta, lower_log, upper_log))


def stage_per_operation_nll(
    prepared: PreparedLikelihood,
    stage: str,
    delta_log_kappa: float,
) -> np.ndarray:
    """Return one stage's masked per-operation NLL, zero outside the mask."""
    if stage not in STAGE_NAMES:
        raise ValueError(f"unknown likelihood stage: {stage}")
    concentration = adjusted_concentration(prepared.concentration, delta_log_kappa)
    nll = -_beta_binomial_log_pmf_validated(
        prepared.successes[stage],
        prepared.trials[stage],
        prepared.probabilities[stage],
        concentration,
    )
    return np.where(prepared.active_mask, nll, 0.0)


def stage_row_scores(
    prepared: PreparedLikelihood,
    stage: str,
    delta_log_kappa: float,
) -> np.ndarray:
    """Average one stage over selected operations within each row."""
    per_operation = stage_per_operation_nll(prepared, stage, delta_log_kappa)
    return np.sum(per_operation, axis=1, dtype=np.float64) / prepared.active_per_row


def score_rows(
    prepared: PreparedLikelihood,
    delta_by_stage: Mapping[str, float],
) -> np.ndarray:
    """Score summed three-stage NLL, averaging operations then equally weighted rows."""
    if set(delta_by_stage) != set(STAGE_NAMES):
        raise ValueError(f"delta_by_stage must specify exactly {STAGE_NAMES}")
    per_operation = np.zeros(prepared.active_mask.shape, dtype=np.float64)
    for stage in STAGE_NAMES:
        np.add(
            per_operation,
            stage_per_operation_nll(prepared, stage, delta_by_stage[stage]),
            out=per_operation,
        )
    return np.sum(per_operation, axis=1, dtype=np.float64) / prepared.active_per_row


def clipping_summary(
    prepared: PreparedLikelihood,
    delta_by_stage: Mapping[str, float],
) -> dict[str, dict[str, float | int]]:
    """Report the share of selected operation kappas clipped at either bound."""
    if set(delta_by_stage) != set(STAGE_NAMES):
        raise ValueError(f"delta_by_stage must specify exactly {STAGE_NAMES}")
    active_count = int(np.count_nonzero(prepared.active_mask))
    log_kappa = np.log(prepared.concentration)
    lower_log, upper_log = np.log(KAPPA_BOUNDS)
    result: dict[str, dict[str, float | int]] = {}
    for stage in STAGE_NAMES:
        delta = float(delta_by_stage[stage])
        if not np.isfinite(delta):
            raise ValueError(f"{stage} delta must be finite")
        unclipped = log_kappa + delta
        clipped_count = int(
            np.count_nonzero(prepared.active_mask & ((unclipped < lower_log) | (unclipped > upper_log)))
        )
        result[stage] = {
            "active_count": active_count,
            "clipped_count": clipped_count,
            "clipping_fraction": clipped_count / active_count,
        }
    return result


def fit_scalar_delta(
    objective: Callable[[float], float],
    bounds: tuple[float, float] = DELTA_BOUNDS_LOG,
) -> dict[str, float | int | bool | str]:
    """Fit one bounded scalar and compare both endpoints with the optimizer result."""
    lower, upper = (float(value) for value in bounds)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("bounds must be finite and strictly increasing")

    def evaluate(value: float) -> float:
        result = float(objective(float(value)))
        if not np.isfinite(result):
            raise ValueError("optimizer objective must return a finite value")
        return result

    optimized = minimize_scalar(
        evaluate,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1.0e-5, "maxiter": 100},
    )
    candidates = (
        (lower, evaluate(lower)),
        (float(optimized.x), evaluate(float(optimized.x))),
        (upper, evaluate(upper)),
    )
    selected_delta, selected_value = min(
        candidates, key=lambda item: (item[1], abs(item[0]), item[0])
    )
    return {
        "delta_log_kappa": float(selected_delta),
        "objective_value": float(selected_value),
        "optimizer_success": bool(optimized.success),
        "optimizer_message": str(optimized.message),
        "optimizer_iterations": int(optimized.nit),
        "optimizer_function_evaluations": int(optimized.nfev),
        "bounds_log": [lower, upper],
        "selected_lower_bound": bool(selected_delta == lower),
        "selected_upper_bound": bool(selected_delta == upper),
        "endpoint_objectives": {
            "lower": float(candidates[0][1]),
            "upper": float(candidates[2][1]),
        },
    }


def gain_summary(gains: object) -> dict[str, float | int | dict[str, float]]:
    """Summarize paired per-row gains as NLL improvement in nats."""
    values = _numeric_array("gains", gains)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("gains must be a nonempty one-dimensional vector")
    quantile_levels = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    quantiles = np.quantile(values, quantile_levels)
    return {
        "n_rows": int(values.size),
        "mean_gain_nats": float(np.mean(values, dtype=np.float64)),
        "worst_loss_nats": float(np.min(values)),
        "gain_quantiles_nats": {
            f"q{int(level * 100):02d}": float(value)
            for level, value in zip(quantile_levels, quantiles)
        },
    }


def group_bootstrap_ci(
    gains: object,
    group_ids: Sequence[object],
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    progress_callback: Callable[[], None] | None = None,
) -> dict[str, float | int | str]:
    """Cluster bootstrap with replacement and a row-weighted mean per draw."""
    values = _numeric_array("gains", gains)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("gains must be a nonempty one-dimensional vector")
    groups = np.asarray([str(value) for value in group_ids], dtype=str)
    if groups.shape != values.shape or np.any(np.char.str_len(groups) == 0):
        raise ValueError("group_ids must provide one nonempty identifier per row")
    if not isinstance(n_resamples, (int, np.integer)) or isinstance(n_resamples, (bool, np.bool_)):
        raise ValueError("n_resamples must be a positive integer")
    if int(n_resamples) <= 0:
        raise ValueError("n_resamples must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")

    _, inverse = np.unique(groups, return_inverse=True)
    group_count = int(np.max(inverse)) + 1
    row_counts = np.bincount(inverse, minlength=group_count).astype(np.float64)
    group_sums = np.bincount(inverse, weights=values, minlength=group_count).astype(np.float64)
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_resamples), dtype=np.float64)
    for index in range(int(n_resamples)):
        selected = rng.integers(0, group_count, size=group_count)
        means[index] = np.sum(group_sums[selected], dtype=np.float64) / np.sum(
            row_counts[selected], dtype=np.float64
        )
        if progress_callback is not None and index % 64 == 0:
            progress_callback()
    lower, upper = np.quantile(means, (0.025, 0.975))
    return {
        "method": "group_resampling_with_row_weighted_mean",
        "resamples": int(n_resamples),
        "seed": int(seed),
        "n_groups": group_count,
        "observed_mean_nats": float(np.mean(values, dtype=np.float64)),
        "ci95_lower_nats": float(lower),
        "ci95_upper_nats": float(upper),
    }


def frozen_decision(
    primary_gain: Mapping[str, object],
    primary_interval: Mapping[str, object],
    reverse_gain: Mapping[str, object],
    thresholds: Mapping[str, object],
    *,
    optimizer_success: bool,
) -> dict[str, object]:
    """Apply the registered primary screen and calibration-to-validation check."""
    expected_keys = {
        "primary_mean_gain_minimum",
        "primary_ci95_lower_minimum",
        "reverse_mean_gain_minimum",
    }
    if set(thresholds) != expected_keys:
        raise ValueError(f"thresholds must specify exactly {sorted(expected_keys)}")
    primary_min = float(thresholds["primary_mean_gain_minimum"])
    ci_min = float(thresholds["primary_ci95_lower_minimum"])
    reverse_min = float(thresholds["reverse_mean_gain_minimum"])
    if not all(np.isfinite(value) for value in (primary_min, ci_min, reverse_min)):
        raise ValueError("decision thresholds must be finite")
    checks = {
        "primary_mean_gain_pass": float(primary_gain["mean_gain_nats"]) >= primary_min,
        "primary_ci95_lower_pass": float(primary_interval["ci95_lower_nats"]) > ci_min,
        "reverse_mean_gain_pass": float(reverse_gain["mean_gain_nats"]) > reverse_min,
        "optimizer_success": bool(optimizer_success),
    }
    passed = all(checks.values())
    return {
        "decision": "nested_stage_calibration_meets_frozen_screen" if passed else "keep_global_shared_dial_no_further_family",
        "checks": checks,
        "thresholds": {
            "primary_mean_gain_minimum": primary_min,
            "primary_ci95_lower_minimum_strict": ci_min,
            "reverse_mean_gain_minimum_strict": reverse_min,
        },
        "next_action": (
            "retain_as_research_only_candidate_pending_separate_registration"
            if passed
            else "retain_global_shared_calibration_and_register_no_further_family"
        ),
    }
