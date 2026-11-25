"""Validate source- and checkpoint-bound classifier update measurements.

This checks an audit artifact's recorded observations.  It does not establish
that its producer or checkpoint is honest, calibrated, or scientifically valid.
Those are separate source/checkpoint review and calibration authority boundaries.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


LEGACY_AUDIT_SCHEMA = "V10ClassifierOutputSliceAuditV3"
PHYSICAL_AUDIT_SCHEMA = "V10ClassifierOutputSliceAuditV4"
SUMMARY_SCHEMA = "V10CalibrationUpdateEvidenceSummaryV2"
_IDENTITY_LENGTHS = {"source_commit": 40, "source_tree_sha": 40, "config_sha": 64}
_OUTPUT_BINDINGS = {
    "R1_re": (("R1_re",), "direct factor-slice measurement"),
    "R1_im": (
        ("R1_re",),
        "derived from the same R1 factor slice; R1 loss covers real and imaginary parts",
    ),
    "W_populations": (
        ("R1_re",),
        "derived from the same R1 factor slice; R1 loss covers populations",
    ),
    "W_coherence_re": (
        ("R1_re",),
        "derived from the same R1 factor slice; R1 loss covers coherence",
    ),
    "W_coherence_im": (
        ("R1_re",),
        "derived from the same R1 factor slice; R1 loss covers coherence",
    ),
    "W_amplitude_variance": (
        ("W_amplitude_variance",),
        "direct variance slice measurement",
    ),
    "W_phase_variance": (("W_phase_variance",), "direct variance slice measurement"),
    "joint_control_covariance": (
        ("W_amplitude_variance", "W_phase_variance"),
        "joint covariance requires both marginal variance slices",
    ),
    "sector_aggregate_posterior": (
        ("sector_aggregate",),
        "K1 aggregate posterior equals R1 trace and uses the native aggregate sector slice",
    ),
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite non-boolean number")
    return float(value)


def _canonical_sha(value: Mapping[str, Any]) -> str:
    payload = {
        key: item for key, item in value.items() if key != "audit_canonical_sha256"
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate output-audit JSON key")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError(f"nonfinite output-audit JSON constant: {value}")


def _expected_model_state(expected_bindings: Mapping[str, str]) -> str:
    values = [
        expected_bindings[key]
        for key in ("final_model_state_sha", "model_state_sha")
        if key in expected_bindings
    ]
    if (
        not values
        or not all(_is_hex(value, 64) for value in values)
        or len(set(values)) != 1
    ):
        raise ValueError("expected bindings require one exact final model-state SHA")
    return values[0]


def _validate_identity(
    audit: Mapping[str, Any], expected_bindings: Mapping[str, str]
) -> tuple[dict[str, str], str]:
    source_identity = audit.get("source_identity")
    if not isinstance(source_identity, Mapping):
        raise ValueError("audit lacks source identity")
    result = {}
    for name, length in _IDENTITY_LENGTHS.items():
        expected = expected_bindings.get(name)
        actual = source_identity.get(name)
        if not _is_hex(expected, length) or actual != expected:
            raise ValueError(f"audit source identity mismatch for {name}")
        result[name] = actual
    state_sha = _expected_model_state(expected_bindings)
    if audit.get("final_model_state_sha") != state_sha:
        raise ValueError("audit final model-state SHA mismatch")
    from . import classifier_output_audit

    producer_sha = _sha256_file(Path(classifier_output_audit.__file__))
    if audit.get("producer_source_sha") != producer_sha:
        raise ValueError(
            "audit producer source SHA does not match classifier_output_audit.py"
        )
    return result, state_sha


def _validate_cells(audit: Mapping[str, Any]) -> tuple[str, ...]:
    registered = audit.get("registered_cells")
    observed = audit.get("audited_cells")
    if not all(
        isinstance(values, list)
        and values == sorted(set(values))
        and values
        and all(isinstance(cell, str) and cell for cell in values)
        for values in (registered, observed)
    ):
        raise ValueError(
            "audit registered and observed cells must be non-empty sorted unique strings"
        )
    if registered != observed:
        raise ValueError("audit registered cells and observed cells differ")
    return tuple(registered)


def _committed_step_ids(audit: Mapping[str, Any]) -> list[int]:
    step_ids = audit.get("committed_step_ids")
    declared_count = audit.get("committed_step_count")
    if (
        not isinstance(step_ids, list)
        or not step_ids
        or any(type(step) is not int or step < 1 for step in step_ids)
    ):
        raise ValueError("audit requires non-empty positive committed step IDs")
    if step_ids != sorted(set(step_ids)) or step_ids != list(
        range(step_ids[0], step_ids[-1] + 1)
    ):
        raise ValueError(
            "audit committed step IDs must be sorted, unique, and contiguous within this execution"
        )
    if type(declared_count) is not int or declared_count != len(step_ids):
        raise ValueError("audit committed step count does not match committed step IDs")
    return step_ids


def _validate_fit_membership(
    audit: Mapping[str, Any],
    expected_row_ids: tuple[str, ...],
    expected_group_ids: tuple[str, ...],
    committed_step_ids: list[int],
) -> dict[str, list[str]]:
    if (
        len(expected_row_ids) != len(expected_group_ids)
        or not expected_row_ids
        or expected_row_ids != tuple(sorted(set(expected_row_ids)))
        or any(not isinstance(value, str) or not value for value in expected_group_ids)
    ):
        raise ValueError("expected fit membership is not a canonical row-to-group map")
    expected = dict(zip(expected_row_ids, expected_group_ids, strict=True))
    observed = audit.get("observed_fit_membership")
    if not isinstance(observed, Mapping) or set(observed) != {
        "row_ids",
        "acquisition_group_ids",
    }:
        raise ValueError("output audit lacks actual committed fit membership")
    observed_rows = observed["row_ids"]
    observed_groups = observed["acquisition_group_ids"]
    if observed_rows != list(expected_row_ids) or observed_groups != list(
        expected_group_ids
    ):
        raise ValueError("actual committed rows differ from the bound fit membership")
    batches = audit.get("committed_batch_memberships")
    if not isinstance(batches, list) or len(batches) != len(committed_step_ids):
        raise ValueError("every committed step must retain its actual batch membership")
    observed_union = {}
    for step, batch in zip(committed_step_ids, batches, strict=True):
        if not isinstance(batch, Mapping) or set(batch) != {
            "global_step",
            "sample_ids",
            "acquisition_group_ids",
        }:
            raise ValueError("committed batch membership has an invalid schema")
        rows = batch["sample_ids"]
        groups = batch["acquisition_group_ids"]
        if (
            batch["global_step"] != step
            or not isinstance(rows, list)
            or not isinstance(groups, list)
            or not rows
            or len(rows) != len(groups)
            or len(rows) != len(set(rows))
        ):
            raise ValueError("committed batch membership is incomplete or misaligned")
        for row_id, group_id in zip(rows, groups, strict=True):
            if expected.get(row_id) != group_id:
                raise ValueError("committed batch contains a row outside the fit membership")
            observed_union[row_id] = group_id
    if observed_union != expected:
        raise ValueError("committed optimizer steps did not cover the exact fit membership")
    return {
        "row_ids": list(expected_row_ids),
        "acquisition_group_ids": list(expected_group_ids),
    }


def _records_by_slice(
    audit: Mapping[str, Any],
    registered_cells: tuple[str, ...],
    committed_step_ids: list[int],
) -> tuple[list[Mapping[str, Any]], int]:
    final_step = audit.get("final_committed_step")
    records = audit.get("records")
    if (
        audit.get("audit_complete") is not True
        or audit.get("final_committed_step_observed") is not True
    ):
        raise ValueError("audit is incomplete")
    if (
        type(final_step) is not int
        or final_step != committed_step_ids[-1]
        or not isinstance(records, list)
        or not records
    ):
        raise ValueError("audit must have a positive final committed step and records")
    steps = []
    for record in records:
        if (
            not isinstance(record, Mapping)
            or type(record.get("global_step")) is not int
            or record["global_step"] < 1
            or record.get("committed_step") is not True
        ):
            raise ValueError("audit records require positive integer committed steps")
        if record.get("cell_id") not in registered_cells:
            raise ValueError("audit record references an unregistered cell")
        if record["global_step"] not in committed_step_ids:
            raise ValueError("audit record is not bound to a committed step")
        steps.append(record["global_step"])
    if (
        steps != sorted(steps)
        or len(steps) != len(set(steps))
        or steps[-1] != final_step
    ):
        raise ValueError(
            "audit records must be unique sorted committed steps ending at the final step"
        )
    return records, final_step


def _validated_target_coverage(
    audit: Mapping[str, Any],
    registered_cells: tuple[str, ...],
    required_outputs: tuple[str, ...],
) -> dict[str, dict[str, dict[str, int | str]]]:
    coverage = audit.get("target_coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("audit lacks committed target coverage")
    result = {}
    for cell_id in registered_cells:
        cell_coverage = coverage.get(cell_id)
        if not isinstance(cell_coverage, Mapping):
            raise ValueError("audit lacks target coverage for a registered cell")
        result[cell_id] = {}
        for output_name in required_outputs:
            row = cell_coverage.get(output_name)
            if not isinstance(row, Mapping) or row.get("unit") != "MASK_ELEMENTS":
                raise ValueError(
                    f"audit target coverage unit is invalid for {output_name}"
                )
            valid = row.get("valid")
            available = row.get("available")
            if (
                type(valid) is not int
                or type(available) is not int
                or valid <= 0
                or available < valid
            ):
                raise ValueError(
                    f"audit lacks positive committed target coverage for {output_name}"
                )
            result[cell_id][output_name] = {
                "valid": valid,
                "available": available,
                "unit": "MASK_ELEMENTS",
            }
    return result


def _validated_physical_objective_coverage(
    audit: Mapping[str, Any],
    registered_cells: tuple[str, ...],
    required_outputs: tuple[str, ...],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """Require actual committed physical-loss support, never raw mask elements."""
    from .published_output_registry import physical_output_training_registry

    static = audit.get("static_target_availability")
    effective = audit.get("effective_objective_coverage")
    if not isinstance(static, Mapping) or not isinstance(effective, Mapping):
        raise ValueError("physical audit lacks static and effective objective coverage")
    registry = physical_output_training_registry()
    declared_bindings = audit.get("objective_bindings")
    if not isinstance(declared_bindings, Mapping):
        raise ValueError("physical audit lacks explicit objective bindings")
    result = {}
    bindings_by_output = {}
    for cell_id in registered_cells:
        static_cell = static.get(cell_id)
        effective_cell = effective.get(cell_id)
        if not isinstance(static_cell, Mapping) or not isinstance(effective_cell, Mapping):
            raise ValueError("physical audit lacks coverage for a registered cell")
        result[cell_id] = {}
        for output_name in required_outputs:
            spec = registry.get(output_name)
            bindings = getattr(spec, "objective_bindings", ()) if spec else ()
            if not bindings:
                raise ValueError(
                    f"physical feedback output lacks explicit objective bindings: {output_name}"
                )
            expected_binding_rows = [
                {
                    "objective_name": binding.objective_name,
                    "authoritative_targets": list(binding.authoritative_targets),
                    "target_masks": list(binding.target_masks),
                    "effective_mask_key": binding.effective_mask_key,
                    "coverage_unit": binding.coverage_unit,
                    "relation": binding.relation,
                    "scope": binding.scope,
                }
                for binding in bindings
            ]
            if declared_bindings.get(output_name) != expected_binding_rows:
                raise ValueError(
                    f"physical audit objective bindings differ for {output_name}"
                )
            static_row = static_cell.get(output_name)
            if (
                not isinstance(static_row, Mapping)
                or static_row.get("unit") != "MASK_ELEMENTS"
                or type(static_row.get("valid")) is not int
                or type(static_row.get("available")) is not int
                or static_row["valid"] <= 0
                or static_row["available"] < static_row["valid"]
            ):
                raise ValueError(
                    f"physical audit lacks positive static target availability for {output_name}"
                )
            rows = []
            for binding in bindings:
                row = effective_cell.get(binding.objective_name)
                if (
                    not isinstance(row, Mapping)
                    or row.get("unit") != binding.coverage_unit
                    or row.get("effective_mask_key") != binding.effective_mask_key
                    or type(row.get("valid")) is not int
                    or type(row.get("available")) is not int
                    or row["valid"] <= 0
                    or row["available"] < row["valid"]
                ):
                    raise ValueError(
                        "physical audit lacks positive effective objective coverage for "
                        f"{output_name}:{binding.objective_name}"
                    )
                rows.append(
                    {
                        "objective_name": binding.objective_name,
                        "effective_mask_key": binding.effective_mask_key,
                        "coverage_unit": binding.coverage_unit,
                        "valid": row["valid"],
                        "available": row["available"],
                    }
                )
            result[cell_id][output_name] = rows
            bindings_by_output[output_name] = rows
    return result, bindings_by_output


def _slice_summary(records: list[Mapping[str, Any]], slice_name: str) -> dict[str, Any]:
    evidence = []
    for record in records:
        outputs = record.get("outputs")
        row = outputs.get(slice_name) if isinstance(outputs, Mapping) else None
        if not isinstance(row, Mapping):
            raise ValueError(f"audit record is missing raw output row for {slice_name}")
        present = row.get("tensor_gradient_present")
        nonzero = row.get("slice_gradient_nonzero")
        norm = _finite_number(
            row.get("slice_gradient_norm"), f"{slice_name}.slice_gradient_norm"
        )
        delta = _finite_number(
            row.get("parameter_delta_norm"), f"{slice_name}.parameter_delta_norm"
        )
        if (
            type(present) is not bool
            or type(nonzero) is not bool
            or present is not True
            or nonzero != (norm > 0.0)
            or norm < 0.0
            or delta < 0.0
        ):
            raise ValueError(f"audit gradient flags are inconsistent for {slice_name}")
        evidence.append((record["global_step"], norm, delta))
    positive = [
        (step, norm, delta)
        for step, norm, delta in evidence
        if norm > 0.0 and delta > 0.0
    ]
    if not positive:
        raise ValueError(
            f"audit lacks positive raw gradient and parameter-delta evidence for {slice_name}"
        )
    return {
        "audit_slice": slice_name,
        "record_count": len(evidence),
        "positive_update_steps": [step for step, _, _ in positive],
        "final_parameter_delta_norm": evidence[-1][2],
        "max_gradient_norm": max(norm for _, norm, _ in evidence),
    }


def validate_output_update_evidence(
    path: str | Path,
    expected_sha: str,
    expected_bindings: Mapping[str, str],
    required_outputs: tuple[str, ...],
    *,
    expected_fit_row_ids: tuple[str, ...] | None = None,
    expected_fit_group_ids: tuple[str, ...] | None = None,
    require_physical_objective_coverage: bool = False,
) -> dict[str, Any]:
    """Return a canonical per-output summary from an immutable raw audit JSON."""
    audit_path = Path(path)
    if not audit_path.is_file() or not _is_hex(expected_sha, 64):
        raise ValueError("output audit bytes do not match the declared SHA-256")
    try:
        payload = audit_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha:
            raise ValueError("output audit bytes do not match the declared SHA-256")
        audit = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("output audit is not valid JSON") from error
    if (
        not isinstance(audit, dict)
        or audit.get("schema") not in {LEGACY_AUDIT_SCHEMA, PHYSICAL_AUDIT_SCHEMA}
        or audit.get("audit_canonical_sha256") != _canonical_sha(audit)
    ):
        raise ValueError("output audit schema or canonical SHA is invalid")
    if (
        not isinstance(expected_bindings, Mapping)
        or not isinstance(required_outputs, tuple)
        or not required_outputs
        or len(set(required_outputs)) != len(required_outputs)
    ):
        raise ValueError("output update evidence arguments are invalid")
    unsupported = [name for name in required_outputs if name not in _OUTPUT_BINDINGS]
    if unsupported:
        raise ValueError(
            f"unsupported required feedback outputs: {sorted(unsupported)}"
        )
    physical_consumer_request = (
        require_physical_objective_coverage
        or "sector_aggregate_posterior" in required_outputs
    )
    if physical_consumer_request:
        from .published_output_registry import physical_feedback_required_outputs

        if required_outputs != physical_feedback_required_outputs:
            raise ValueError("physical calibration requires the complete consumer output contract")
    source_identity, model_state_sha = _validate_identity(audit, expected_bindings)
    registered_cells = _validate_cells(audit)
    committed_step_ids = _committed_step_ids(audit)
    if (expected_fit_row_ids is None) != (expected_fit_group_ids is None):
        raise ValueError("fit row and acquisition-group membership must be supplied together")
    fit_membership = (
        None
        if expected_fit_row_ids is None
        else _validate_fit_membership(
            audit,
            expected_fit_row_ids,
            expected_fit_group_ids,
            committed_step_ids,
        )
    )
    records, final_step = _records_by_slice(audit, registered_cells, committed_step_ids)
    is_physical = audit["schema"] == PHYSICAL_AUDIT_SCHEMA
    if physical_consumer_request and not is_physical:
        raise ValueError("calibration admission requires a physical V4 objective audit")
    if is_physical:
        target_coverage, bindings_by_output = _validated_physical_objective_coverage(
            audit, registered_cells, required_outputs
        )
    else:
        target_coverage = _validated_target_coverage(
            audit, registered_cells, required_outputs
        )
        bindings_by_output = {}
    slice_cache = {}
    output_summary = {}
    for output_name in required_outputs:
        slices, alias_reason = _OUTPUT_BINDINGS[output_name]
        summaries = []
        for slice_name in slices:
            slice_cache.setdefault(slice_name, _slice_summary(records, slice_name))
            summaries.append(slice_cache[slice_name])
        output_summary[output_name] = {
            "audit_slices": list(slices),
            "binding_reason": alias_reason,
            "slice_evidence": summaries,
            **(
                {"objective_bindings": bindings_by_output[output_name]}
                if is_physical
                else {}
            ),
        }
    return {
        "schema": "V10CalibrationUpdateEvidenceSummaryV3" if physical_consumer_request else SUMMARY_SCHEMA,
        "classifier_output_audit_sha": expected_sha,
        "source_identity": source_identity,
        "final_model_state_sha": model_state_sha,
        "registered_cells": list(registered_cells),
        "actual_fit_membership": fit_membership,
        "committed_step_count": len(committed_step_ids),
        "audited_committed_step_count": len(records),
        "final_committed_step": final_step,
        **(
            {"effective_objective_coverage": target_coverage}
            if is_physical
            else {"target_coverage": target_coverage}
        ),
        "per_output": output_summary,
        "boundary": "Raw source/checkpoint-bound measurements require separate external calibration review and do not alone prove calibration or scientific authority",
    }
