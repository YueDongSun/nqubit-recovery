"""PIT adapter for the unequal-concentration research overlay.

The adapter consumes a validated native point artifact, its separate unequal
concentration overlay, and explicitly supplied observed four-cell rows. It
reuses the registered native beta-binomial CDF kernel without executing a
model, fitting a parameter, or changing the legacy shared-concentration PIT.
"""
from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any, Mapping, Sequence

import torch

from .contracts import canonical_sha256, require_sha256
from .effective_inference_artifact import Effective_point_inference
from .effective_point_values import numerical_array, probability_names
from .native_predictive_pit import (
    BRANCH_NAMES, Native_pit_config, beta_binomial_cdf_interval,
    deterministic_pit_uniform, native_pit_numerical_domain,
)
from .unequal_concentration_point import (
    Unequal_concentration_point_overlay,
    validate_unequal_concentration_point_artifact,
)


unequal_pit_schema = "TrackBUnequalConcentrationConditionalPitV1"
unequal_pit_plan_schema = "TrackBUnequalConcentrationPitPlanV1"
pit_scope = "FROZEN_NATIVE_POINT_WITH_UNEQUAL_CONCENTRATION_OVERLAY"
observation_fields = frozenset({"sample_id", "four_cell_counts"})
record_fields = frozenset({
    "sample_id", "operation_id", "branch", "success", "trials",
    "probability", "concentration", "informative",
})
plan_fields = frozenset({
    "schema", "scope", "config", "numerical_domain",
    "base_point_artifact_sha", "overlay_file_sha", "overlay_theta_sha",
    "observation_sha", "plan_sha", "query_branches", "informative_branches",
    "cdf_pmf_terms", "records", "cdf_evaluated", "scientific_gate_issued",
})


def _checked_observations(
    base_payload: Mapping[str, Any], observations: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, tuple[tuple[int, int, int, int], ...]], str]:
    if type(observations) is not list or len(observations) != len(base_payload["rows"]):
        raise ValueError("unequal concentration PIT observations must cover every point row")
    observed: dict[str, tuple[tuple[int, int, int, int], ...]] = {}
    serialized = []
    for row, item in zip(base_payload["rows"], observations, strict=True):
        if type(item) is not dict or set(item) != observation_fields or item["sample_id"] != row["sample_id"]:
            raise ValueError("unequal concentration PIT observation identity differs")
        raw_counts = item["four_cell_counts"]
        operation_count = len(row["operation_ids"])
        native_read_count = row["values"]["native_read_count"]
        if type(native_read_count) is not int or native_read_count < 0:
            raise ValueError("unequal concentration PIT parent READ count is invalid")
        if type(raw_counts) is not list or len(raw_counts) != operation_count:
            raise ValueError("unequal concentration PIT count shape differs")
        counts = []
        for cells in raw_counts:
            if type(cells) is not list or len(cells) != 4 \
                    or any(type(value) is not int or value < 0 for value in cells):
                raise ValueError("unequal concentration PIT counts must be nonnegative integer cells")
            counts.append(tuple(cells))
        for cells, available, query in zip(
            counts, row["operation_mask"], row["query_mask"], strict=True,
        ):
            if type(available) is not bool or type(query) is not bool:
                raise ValueError("unequal concentration PIT parent masks are invalid")
            if query and not available:
                raise ValueError("unequal concentration PIT query selects unavailable operation")
            if query and sum(cells) != native_read_count:
                raise ValueError(
                    "unequal concentration PIT observed total differs from bound native READ"
                )
        if item["sample_id"] in observed:
            raise ValueError("unequal concentration PIT sample identities must be unique")
        observed[item["sample_id"]] = tuple(counts)
        serialized.append({"sample_id": item["sample_id"], "four_cell_counts": raw_counts})
    return observed, canonical_sha256(serialized)


def _validate_pit_record_domain(
    success: int, trials: int, probability: float, concentration: float,
    config: Native_pit_config,
) -> None:
    domain = native_pit_numerical_domain()
    if (
        type(success) is not int or type(trials) is not int
        or not 0 <= success <= trials <= int(domain["maximum_trials"])
        or trials > 2**53 - 1
    ):
        raise ValueError("unequal concentration PIT counts exceed the registered domain")
    if not (
        float(domain["minimum_probability"]) <= probability <= float(domain["maximum_probability"])
        and float(domain["minimum_concentration"]) <= concentration <= float(domain["maximum_concentration"])
    ):
        raise ValueError("unequal concentration PIT input is outside the registered numerical domain")
    required = 0 if trials == 0 else 1 + min(success, trials - success)
    if required > config.max_terms_per_observation:
        raise ValueError("unequal concentration PIT per-observation budget exceeded")


def _row_records(
    base_row: Mapping[str, Any], overlay_row: Mapping[str, Any],
    counts: tuple[tuple[int, int, int, int], ...], config: Native_pit_config,
) -> list[dict[str, Any]]:
    values = base_row["values"]
    probabilities = {
        name: numerical_array(values[name], (len(base_row["operation_ids"]),), name)
        for name in probability_names
    }
    concentration = numerical_array(
        overlay_row["values"]["concentration_by_branch"],
        (len(base_row["operation_ids"]), 3), "concentration_by_branch",
    )
    records = []
    for operation_index, (operation_id, available, query) in enumerate(zip(
        base_row["operation_ids"], base_row["operation_mask"], base_row["query_mask"], strict=True,
    )):
        if type(available) is not bool or type(query) is not bool or query and not available:
            raise ValueError("unequal concentration PIT masks are invalid")
        if not query:
            continue
        n00, n10, n01, n11 = counts[operation_index]
        signal = n10 + n11
        no_signal = n00 + n01
        branches = ((signal, signal + no_signal), (n11, signal), (n01, no_signal))
        for branch_index, (success, trials) in enumerate(branches):
            probability = float(probabilities[probability_names[branch_index]][operation_index])
            kappa = float(concentration[operation_index, branch_index])
            if not math.isfinite(probability) or not 0.0 < probability < 1.0:
                raise ValueError("unequal concentration PIT probability is outside the native interior")
            if not math.isfinite(kappa) or kappa <= 0.0:
                raise ValueError("unequal concentration PIT concentration is invalid")
            _validate_pit_record_domain(success, trials, probability, kappa, config)
            records.append({
                "sample_id": base_row["sample_id"], "operation_id": operation_id,
                "branch": BRANCH_NAMES[branch_index], "success": success,
                "trials": trials, "probability": probability,
                "concentration": kappa, "informative": trials > 0,
            })
    return records


def _plan_identity(payload: Mapping[str, Any]) -> str:
    identity = {name: payload[name] for name in plan_fields if name != "plan_sha"}
    return canonical_sha256(identity)


def build_unequal_concentration_pit_plan(
    base_artifact: Effective_point_inference,
    overlay_artifact: Unequal_concentration_point_overlay,
    observations: list[Mapping[str, Any]], *,
    config: Native_pit_config = Native_pit_config(),
) -> dict[str, Any]:
    """Create a source-bound PIT plan without evaluating any CDF."""
    if type(base_artifact) is not Effective_point_inference \
            or type(overlay_artifact) is not Unequal_concentration_point_overlay:
        raise TypeError("unequal concentration PIT requires typed native and overlay artifacts")
    config.validate()
    base_payload = base_artifact.payload()
    overlay_payload = overlay_artifact.payload()
    validate_unequal_concentration_point_artifact(overlay_payload, base_payload=base_payload)
    observed, observation_sha = _checked_observations(base_payload, observations)
    records = []
    for base_row, overlay_row in zip(base_payload["rows"], overlay_payload["rows"], strict=True):
        records.extend(_row_records(base_row, overlay_row, observed[base_row["sample_id"]], config))
    if not records:
        raise ValueError("unequal concentration PIT selected zero query branches")
    terms = sum(0 if not item["informative"] else 1 + min(item["success"], item["trials"] - item["success"])
                for item in records)
    if terms > config.max_terms_per_batch:
        raise ValueError("unequal concentration PIT batch budget exceeded")
    payload = {
        "schema": unequal_pit_plan_schema, "scope": pit_scope,
        "config": asdict(config), "numerical_domain": native_pit_numerical_domain(),
        "base_point_artifact_sha": base_artifact.file_sha,
        "overlay_file_sha": overlay_artifact.file_sha,
        "overlay_theta_sha": overlay_payload["theta_sha"],
        "observation_sha": observation_sha, "plan_sha": "0" * 64,
        "query_branches": len(records),
        "informative_branches": sum(item["informative"] for item in records),
        "cdf_pmf_terms": terms, "records": records, "cdf_evaluated": False,
        "scientific_gate_issued": False,
    }
    payload["plan_sha"] = _plan_identity(payload)
    return payload


def validate_unequal_concentration_pit_plan(plan: Mapping[str, Any]) -> None:
    """Validate the immutable preflight plan before any CDF work."""
    if type(plan) is not dict or set(plan) != plan_fields \
            or plan["schema"] != unequal_pit_plan_schema or plan["scope"] != pit_scope:
        raise ValueError("unequal concentration PIT plan schema or fields differ")
    if type(plan["config"]) is not dict:
        raise ValueError("unequal concentration PIT plan configuration is invalid")
    try:
        config = Native_pit_config(**plan["config"])
        config.validate()
    except (TypeError, ValueError) as error:
        raise ValueError("unequal concentration PIT plan configuration is invalid") from error
    if plan["numerical_domain"] != native_pit_numerical_domain() \
            or plan["cdf_evaluated"] is not False or plan["scientific_gate_issued"] is not False:
        raise ValueError("unequal concentration PIT plan domain or authority fields differ")
    for name in ("base_point_artifact_sha", "overlay_file_sha", "overlay_theta_sha", "observation_sha", "plan_sha"):
        require_sha256(plan[name], name)
    records = plan["records"]
    if type(records) is not list or not records:
        raise ValueError("unequal concentration PIT plan has no records")
    if plan["query_branches"] != len(records) \
            or plan["informative_branches"] != sum(bool(item.get("informative")) for item in records):
        raise ValueError("unequal concentration PIT plan counts are inconsistent")
    terms = 0
    for record in records:
        if type(record) is not dict or set(record) != record_fields:
            raise ValueError("unequal concentration PIT record fields differ")
        if type(record["sample_id"]) is not str or not record["sample_id"] \
                or type(record["operation_id"]) is not str or not record["operation_id"] \
                or record["branch"] not in BRANCH_NAMES:
            raise ValueError("unequal concentration PIT record identity is invalid")
        success, trials = record["success"], record["trials"]
        if type(success) is not int or type(trials) is not int:
            raise ValueError("unequal concentration PIT record counts are invalid")
        probability, concentration = record["probability"], record["concentration"]
        if type(probability) not in (int, float) or not math.isfinite(probability) \
                or type(concentration) not in (int, float) or not math.isfinite(concentration):
            raise ValueError("unequal concentration PIT record numeric inputs are invalid")
        _validate_pit_record_domain(success, trials, probability, concentration, config)
        if record["informative"] is not (trials > 0):
            raise ValueError("unequal concentration PIT informative flag is invalid")
        terms += 0 if trials == 0 else 1 + min(success, trials - success)
    if plan["cdf_pmf_terms"] != terms or terms > config.max_terms_per_batch:
        raise ValueError("unequal concentration PIT plan PMF budget is inconsistent")
    if _plan_identity(plan) != plan["plan_sha"]:
        raise ValueError("unequal concentration PIT plan identity differs")


def execute_unequal_concentration_pit_plan(
    plan: Mapping[str, Any], *, expected_plan_sha: str | None = None,
) -> dict[str, Any]:
    """Evaluate only a previously validated plan with the native CDF kernel."""
    validate_unequal_concentration_pit_plan(plan)
    if expected_plan_sha is not None and plan["plan_sha"] != expected_plan_sha:
        raise ValueError("unequal concentration PIT plan differs from the expected identity")
    config = Native_pit_config(**plan["config"])
    records = []
    evaluated_terms = 0
    for record in plan["records"]:
        interval = beta_binomial_cdf_interval(
            record["success"], record["trials"], record["probability"],
            record["concentration"], config=config,
        )
        expected_terms = 0 if record["trials"] == 0 else 1 + min(
            record["success"], record["trials"] - record["success"],
        )
        if interval["pmf_terms"] != expected_terms:
            raise ArithmeticError("unequal concentration PIT PMF work differs from its plan")
        evaluated_terms += interval["pmf_terms"]
        uniform = deterministic_pit_uniform(
            record["sample_id"], record["operation_id"], record["branch"], config.seed,
        )
        pit = None if not interval["informative"] else interval["left"] + uniform * (
            interval["right"] - interval["left"]
        )
        records.append({**record, "pit": pit, **interval})
    return {
        "schema": unequal_pit_schema, "scope": pit_scope,
        "config": asdict(config), "numerical_domain": native_pit_numerical_domain(),
        "base_point_artifact_sha": plan["base_point_artifact_sha"],
        "overlay_file_sha": plan["overlay_file_sha"],
        "overlay_theta_sha": plan["overlay_theta_sha"],
        "observation_sha": plan["observation_sha"], "plan_sha": plan["plan_sha"],
        "query_branches": len(records), "informative_branches": sum(item["informative"] for item in records),
        "planned_pmf_terms": plan["cdf_pmf_terms"], "actual_pmf_terms": evaluated_terms,
        "records": records, "cdf_evaluated": True, "scientific_gate_issued": False,
    }
