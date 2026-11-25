"""Randomized PIT for the native nested beta-binomial observation law.

CDFs sum the existing native PMF, not a Gaussian or binomial approximation.
Resource bounds are explicit; exceeding them fails instead of dropping hard
observations. Zero-trial conditional branches carry no calibration evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Mapping

import numpy as np
import torch

from ml_recon.physics_hybrid_v8.fpga_joint_observation import _beta_binomial_log_probability
from .losses import native_masked_four_cell_nll
from .contracts import canonical_sha256


BRANCH_NAMES = ("signal_given_read", "coin_given_signal", "idler_given_no_signal")
PIT_CENTRAL_LEVELS = (0.50, 0.80, 0.90, 0.95)
MAX_EXACT_FLOAT_INTEGER = 2**53 - 1
MAX_VALIDATED_PIT_TRIALS = 1000000
MIN_VALIDATED_PIT_PROBABILITY = 1e-9
MIN_VALIDATED_PIT_CONCENTRATION = 5 * math.exp(-12)
MAX_VALIDATED_PIT_CONCENTRATION = 5 * math.exp(12)
NUMERICAL_DOMAIN_SCHEMA = "TrackBNativePitBoundedLgammaDomainV1"


def _validate_pit_numerical_domain(trials: int, probability: float, concentration: float) -> None:
    """Integer exactness alone does not qualify subtractive log-gamma CDFs."""
    if trials > MAX_VALIDATED_PIT_TRIALS \
            or not MIN_VALIDATED_PIT_PROBABILITY <= probability <= 1 - MIN_VALIDATED_PIT_PROBABILITY \
            or not MIN_VALIDATED_PIT_CONCENTRATION <= concentration <= MAX_VALIDATED_PIT_CONCENTRATION:
        raise ValueError("native PIT input is outside the registered numerical domain")


def native_pit_numerical_domain() -> dict[str, Any]:
    """Publish the fixed tested evaluation domain, not a microscopic contract."""
    return {
        "schema": NUMERICAL_DOMAIN_SCHEMA,
        "maximum_trials": MAX_VALIDATED_PIT_TRIALS,
        "minimum_probability": MIN_VALIDATED_PIT_PROBABILITY,
        "maximum_probability": 1 - MIN_VALIDATED_PIT_PROBABILITY,
        "minimum_concentration": MIN_VALIDATED_PIT_CONCENTRATION,
        "maximum_concentration": MAX_VALIDATED_PIT_CONCENTRATION,
        "qualification": "BOUNDARY_AND_INDEPENDENT_REFERENCE_REGRESSION_NOT_A_FORMAL_ERROR_BOUND",
    }


@dataclass(frozen=True)
class Native_pit_config:
    seed: int = 20261001
    max_terms_per_observation: int = 2000000
    max_terms_per_batch: int = 50000000
    chunk_terms: int = 8192
    cdf_roundoff_tolerance: float = 1e-8
    schema: str = "TrackBNativeConditionalPitConfigV1"

    def validate(self) -> None:
        if self.schema != "TrackBNativeConditionalPitConfigV1" or type(self.seed) is not int:
            raise ValueError("invalid native PIT schema or seed")
        if any(type(value) is not int or value <= 0 for value in (
            self.max_terms_per_observation, self.max_terms_per_batch, self.chunk_terms,
        )) or self.chunk_terms > 65536:
            raise ValueError("invalid native PIT resource budget")
        if self.cdf_roundoff_tolerance != 1e-8:
            raise ValueError("native PIT uses the registered explicit roundoff tolerance 1e-8")


def deterministic_pit_uniform(sample_id: str, operation_id: str, branch: str, seed: int) -> float:
    """An open-interval uniform coordinate shared across candidate comparisons."""
    if any(type(value) is not str or not value for value in (sample_id, operation_id, branch)) \
            or branch not in BRANCH_NAMES or type(seed) is not int:
        raise ValueError("PIT randomization requires explicit stable identities")
    payload = f"TrackBNativePitUniformV1|{seed}|{len(sample_id)}:{sample_id}|{len(operation_id)}:{operation_id}|{branch}"
    # 52 bits plus one half are representable and never round to either endpoint.
    word = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") >> 12
    return (word + 0.5) / 2**52


@torch.no_grad()
def beta_binomial_cdf_interval(
    success: int, trials: int, probability: float, concentration: float, *,
    config: Native_pit_config = Native_pit_config(),
) -> dict[str, Any]:
    """Return [F(k-1), F(k)] by bounded summation of the shorter count tail.

Floating-point boundary corrections below the registered tolerance are
reported explicitly. The observation and predictive probabilities are never
clipped. CUDA is not used, even if the caller has a different default device.
"""
    config.validate()
    if type(success) is not int or type(trials) is not int or not 0 <= success <= trials <= MAX_EXACT_FLOAT_INTEGER:
        raise ValueError("PIT counts must be exact supported nonnegative integers")
    if type(probability) not in (int, float) or not math.isfinite(probability) or not 0 < probability < 1 \
            or type(concentration) not in (int, float) or not math.isfinite(concentration) or concentration <= 0:
        raise ValueError("PIT probabilities/concentration must be finite in the native interior")
    _validate_pit_numerical_domain(trials, probability, concentration)
    if trials == 0:
        return {"left": 0.0, "right": 1.0, "pmf": 1.0, "pmf_terms": 0,
                "informative": False, "roundoff_adjustment": 0.0, "pmf_underflow": False}
    tail_terms = min(success, trials - success)
    if tail_terms + 1 > config.max_terms_per_observation:
        raise ValueError("native PIT per-observation CDF budget exceeded")
    reflected = success > trials // 2
    dtype = torch.float64
    n = torch.tensor(float(trials), dtype=dtype, device="cpu")
    p = torch.tensor(float(probability), dtype=dtype, device="cpu")
    kappa = torch.tensor(float(concentration), dtype=dtype, device="cpu")
    k = torch.tensor(float(success), dtype=dtype, device="cpu")
    log_mass = _beta_binomial_log_probability(k, n, p, kappa)
    evaluated_terms = k.numel()
    mass = float(log_mass.exp())
    partials = []
    tail_probability = 1 - p if reflected else p
    for start in range(0, tail_terms, config.chunk_terms):
        indices = torch.arange(start, min(tail_terms, start + config.chunk_terms), dtype=dtype, device="cpu")
        log_terms = _beta_binomial_log_probability(indices, n, tail_probability, kappa)
        evaluated_terms += indices.numel()
        partials.append(float(torch.logsumexp(log_terms, 0).exp()))
    tail = math.fsum(partials)
    if reflected:
        right = 1 - tail
        left = right - mass
    else:
        left = tail
        right = left + mass
    tolerance = config.cdf_roundoff_tolerance
    if not all(math.isfinite(value) for value in (left, right, mass)) \
            or left < -tolerance or right > 1 + tolerance or mass < 0 or left > right:
        raise ArithmeticError("native PIT CDF failed numerical probability support")
    bounded_left = min(1.0, max(0.0, left))
    bounded_right = min(1.0, max(0.0, right))
    return {
        "left": bounded_left, "right": bounded_right, "pmf": mass,
        "pmf_terms": evaluated_terms, "informative": True,
        "roundoff_adjustment": max(abs(bounded_left - left), abs(bounded_right - right)),
        "pmf_underflow": mass == 0,
    }


@torch.no_grad()
def _plan_native_pit_work(
    prediction: Mapping[str, Any], batch: Mapping[str, Any], *,
    config: Native_pit_config = Native_pit_config(),
) -> tuple[list[tuple[int, int, int, int, int]], int]:
    """Validate complete query support and cost before any CDF calculation."""
    config.validate()
    used = [batch[name] for name in ("four_cell_counts", "query_mask", "operation_mask")]
    used.extend(prediction[name] for name in (
        "signal_probability", "coin_given_signal_probability", "idler_given_no_signal_probability", "four_cell_concentration",
    ))
    if any(not isinstance(value, torch.Tensor) or value.device.type != "cpu" for value in used):
        raise ValueError("native PIT accepts CPU tensors only")
    query_mask, operation_mask = batch["query_mask"], batch["operation_mask"]
    if query_mask.dtype != torch.bool or operation_mask.dtype != torch.bool or query_mask.shape != operation_mask.shape:
        raise ValueError("native PIT masks must be aligned booleans")
    if bool(torch.any(query_mask & ~operation_mask)):
        raise ValueError("native PIT query mask selects an unavailable operation")
    native_masked_four_cell_nll(prediction, batch)  # Preserve the native support contract.
    counts = batch["four_cell_counts"]
    query = batch["query_mask"] & batch["operation_mask"]
    sample_ids, operation_ids = batch["sample_ids"], batch["operation_ids"]
    if len(sample_ids) != counts.shape[0] or len(set(sample_ids)) != len(sample_ids) \
            or len(operation_ids) != counts.shape[1] or len(set(operation_ids)) != len(operation_ids):
        raise ValueError("PIT row/operation identities must be aligned and unique")
    probability_names = ("signal_probability", "coin_given_signal_probability", "idler_given_no_signal_probability")
    work = []
    terms = 0
    for row, operation in torch.nonzero(query, as_tuple=False).tolist():
        n00, n10, n01, n11 = (int(value) for value in counts[row, operation].tolist())
        signal = n10 + n11
        no_signal = n00 + n01
        branches = ((signal, signal + no_signal), (n11, signal), (n01, no_signal))
        for branch_index, (success, trials) in enumerate(branches):
            _validate_pit_numerical_domain(
                trials, float(prediction[probability_names[branch_index]][row, operation]),
                float(prediction["four_cell_concentration"][row]),
            )
            required = 0 if trials == 0 else 1 + min(success, trials - success)
            if required > config.max_terms_per_observation:
                raise ValueError("native PIT per-observation CDF budget exceeded")
            terms += required
            work.append((row, operation, branch_index, success, trials))
    if terms > config.max_terms_per_batch:
        raise ValueError("native PIT batch CDF budget exceeded; no observations were dropped")
    return work, terms


def _native_pit_plan(work, terms, prediction, batch, config) -> dict[str, Any]:
    names = ("signal_probability", "coin_given_signal_probability", "idler_given_no_signal_probability")
    rows = [(batch["sample_ids"][row], batch["operation_ids"][operation], BRANCH_NAMES[branch], success, trials,
             float(prediction[names[branch]][row, operation]), float(prediction["four_cell_concentration"][row]))
            for row, operation, branch, success, trials in work]
    identity = {"schema": "TrackBNativePitNumericalPlanV1", "config": asdict(config),
                "numerical_domain": native_pit_numerical_domain(), "query_inputs": rows}
    ranges = {}
    for branch in BRANCH_NAMES:
        selected = [record for record in rows if record[2] == branch]
        ranges[branch] = {name: [min(record[index] for record in selected), max(record[index] for record in selected)]
                          for name, index in (("success", 3), ("trials", 4), ("probability", 5), ("concentration", 6))}
    return {"schema": "TrackBNativePitCostPreflightV1", "plan_sha": canonical_sha256(identity),
            "config": asdict(config), "numerical_domain": native_pit_numerical_domain(),
            "query_branches": len(work), "informative_branches": sum(item[4] > 0 for item in work),
            "cdf_pmf_terms": terms, "numerical_ranges": ranges,
            "cdf_evaluated": False, "scientific_gate_issued": False,
            "cost_scope": "CDF_PMF_TERMS_EXCLUDING_FORWARD_AND_NATIVE_SUPPORT_NLL"}


@torch.no_grad()
def plan_native_conditional_pit(prediction: Mapping[str, Any], batch: Mapping[str, Any], *,
                                config: Native_pit_config = Native_pit_config()) -> dict[str, Any]:
    """Check native support/domain and exact summation work; never compute CDFs."""
    work, terms = _plan_native_pit_work(prediction, batch, config=config)
    return _native_pit_plan(work, terms, prediction, batch, config)


@torch.no_grad()
def native_conditional_pit(
    prediction: Mapping[str, Any], batch: Mapping[str, Any], *,
    config: Native_pit_config = Native_pit_config(), expected_plan_sha: str | None = None,
) -> dict[str, Any]:
    """Execute only the validated native query plan, without model input changes."""
    work, terms = _plan_native_pit_work(prediction, batch, config=config)
    plan = _native_pit_plan(work, terms, prediction, batch, config)
    if expected_plan_sha is not None and plan["plan_sha"] != expected_plan_sha:
        raise ValueError("native PIT numerical inputs differ from the expected plan")
    sample_ids, operation_ids = batch["sample_ids"], batch["operation_ids"]
    probability_names = ("signal_probability", "coin_given_signal_probability", "idler_given_no_signal_probability")
    records = []
    evaluated_terms = 0
    for row, operation, branch_index, success, trials in work:
        branch = BRANCH_NAMES[branch_index]
        interval = beta_binomial_cdf_interval(
            success, trials, float(prediction[probability_names[branch_index]][row, operation]),
            float(prediction["four_cell_concentration"][row]), config=config,
        )
        actual_terms = interval["pmf_terms"]
        expected_terms = 0 if trials == 0 else 1 + min(success, trials - success)
        if type(actual_terms) is not int or actual_terms != expected_terms:
            raise ArithmeticError("native PIT evaluated PMF work differs from the query plan")
        evaluated_terms += actual_terms
        uniform = deterministic_pit_uniform(sample_ids[row], operation_ids[operation], branch, config.seed)
        value = None if not interval["informative"] else interval["left"] + uniform * (interval["right"] - interval["left"])
        records.append({"sample_id": sample_ids[row], "operation_id": operation_ids[operation], "branch": branch,
                        "pit": value, **interval})
    return {"schema": "TrackBNativeConditionalPitV1", "config": asdict(config),
            "plan_sha": plan["plan_sha"],
            "numerical_domain": native_pit_numerical_domain(),
            "method": "NATIVE_PMF_SHORTER_TAIL_SUM", "pmf_terms": evaluated_terms,
            "planned_pmf_terms": terms, "actual_pmf_terms": evaluated_terms,
            "records": records, "scientific_gate_issued": False}


def nonrandomized_pit_summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Integrate uniform randomization inside resolved native CDF jumps.

    This is the discrete integrated-PIT diagnostic, not PMF renormalization.
    An unresolved jump makes the entire summary unavailable, never silently
    removes a difficult observation. No model fit or calibration Gate follows.
    """
    if any(type(record["informative"]) is not bool for record in records):
        raise ValueError("PIT information flags must be strict booleans")
    valid = [record for record in records if record["informative"]]
    intervals = []
    for record in valid:
        left, right = record["left"], record["right"]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in (left, right)) or not 0 <= left <= right <= 1:
            raise ValueError("integrated PIT requires finite ordered CDF endpoints")
        intervals.append((left, right))
    unresolved = sum(left == right for left, right in intervals)
    result = {"method": "CONDITIONAL_CDF_JUMP_RANDOMIZATION_INTEGRATED_OUT",
              "informative_observations": len(valid), "zero_trial_branches": len(records)-len(valid),
              "unresolved_jumps": unresolved, "iid_uniformity_p_value_issued": False,
              "bin_fractions": None, "mean_pit": None, "nominal_central_90_fraction": None,
              "central_coverage_and_tails": None}
    if not valid or unresolved:
        return {**result, "status": "NO_INFORMATIVE_OBSERVATIONS" if not valid else "UNRESOLVED_CDF_JUMPS"}
    edges = np.linspace(0, 1, 11).tolist()
    fractions = [math.fsum(max(0.0, min(right, upper)-max(left, lower))/(right-left)
                           for left, right in intervals) / len(valid)
                 for lower, upper in zip(edges[:-1], edges[1:], strict=True)]
    if not math.isclose(math.fsum(fractions), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ArithmeticError("integrated PIT bin fractions do not sum to one")
    coverage = {}
    for level in PIT_CENTRAL_LEVELS:
        lower, upper = (1-level)/2, (1+level)/2
        masses = [math.fsum(max(0.0, min(right, high)-max(left, low))/(right-left)
                           for left, right in intervals)/len(valid)
                  for low, high in ((0.0, lower), (lower, upper), (upper, 1.0))]
        coverage[str(int(round(level*100)))] = {
            "nominal": level, "lower_tail": masses[0], "central": masses[1], "upper_tail": masses[2],
        }
    return {**result, "status": "DESCRIPTIVE_ONLY", "bin_fractions": fractions,
            "central_coverage_and_tails": coverage,
            "mean_pit": math.fsum((left+right)/2 for left, right in intervals)/len(valid),
            "nominal_central_90_fraction": math.fsum(max(0.0, min(right, 0.95)-max(left, 0.05))/(right-left)
                                                    for left, right in intervals)/len(valid)}


def summarize_native_pit(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Descriptive per-branch summaries; correlated operations are not IID trials."""
    result = {}
    for branch in BRANCH_NAMES:
        selected = [record for record in records if record["branch"] == branch]
        valid = [record for record in selected if record["informative"]]
        values = np.asarray([record["pit"] for record in valid], dtype=np.float64)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("PIT summary values must be finite probabilities")
        result[branch] = {
            "informative_observations": len(valid), "zero_trial_branches": len(selected) - len(valid),
            "mean_pit": None if not len(valid) else float(values.mean()),
            "pit_bin_counts": np.histogram(values, bins=np.linspace(0, 1, 11))[0].tolist(),
            "nominal_central_90_pit_fraction": None if not len(valid) else float(np.mean((values >= 0.05) & (values <= 0.95))),
            "cdf_adjustment_count": sum(record["roundoff_adjustment"] > 0 for record in valid),
            "maximum_cdf_adjustment": max((record["roundoff_adjustment"] for record in valid), default=0.0),
            "pmf_underflow_count": sum(record["pmf_underflow"] for record in valid),
            "iid_uniformity_p_value_issued": False,
            "nonrandomized": nonrandomized_pit_summary(selected),
        }
    return result
