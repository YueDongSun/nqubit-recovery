"""State/view identity for additive native ensemble diagnostics.

Population order and batch positions are provenance, not state identity.
This module pairs only the same admitted sample, session and actual view.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .contracts import canonical_sha256, require_git_oid, require_sha256
from .effective_inference_artifact import Effective_point_inference, point_row_fields
from .effective_inference_context import Effective_inference_config
from .effective_point_values import _validate_psd
from .effective_family_context_v3 import MODEL_SCHEMA as family_model_schema
from .effective_observation_v2 import MODEL_SCHEMA as effective_model_schema


state_context_schema = "TrackBNativeStateViewContextV1"
context_fields = frozenset({
    "schema", "system_type", "model_schema", "model_config_sha", "model_kind",
    "point_reader_commit", "point_reader_tree", "model_producer_commit", "model_producer_tree",
    "training_source_commit", "training_source_tree",
    "admission_manifest_sha", "outer_fold_sha", "outer_role", "selected_row_identity_sha",
    "training_row_identity_sha", "operation_metadata_sha", "query_recipe", "population_row_count",
    "sample_id", "group_id", "row_content_sha", "source_role", "session_index",
    "operation_ids", "operation_families", "cycle_indices", "operation_mask",
    "visible_operation_mask", "query_mask", "native_read_count", "prepared_prior_available",
    "prepared_prior_real", "prepared_prior_imag",
})
view_fields = ("operation_ids", "operation_families", "cycle_indices", "operation_mask",
               "visible_operation_mask", "query_mask")


def validate_native_state_context(context: dict[str, Any]) -> None:
    if type(context) is not dict or set(context) != context_fields or context["schema"] != state_context_schema:
        raise ValueError("Native state context schema/fields differ")
    if context["system_type"] not in {"3qubit", "4qubit"} \
            or context["model_schema"] not in {family_model_schema, effective_model_schema} \
            or context["model_kind"] not in {"EFFECTIVE_CPU_CHECKPOINT", "FAMILY_REFINEMENT_OVERLAY"} \
            or context["outer_role"] not in {"validation", "calibration"} or context["source_role"] != "train":
        raise ValueError("Native state context requires registered non-training inference")
    for name in ("point_reader_commit", "point_reader_tree", "model_producer_commit", "model_producer_tree",
                 "training_source_commit", "training_source_tree"):
        require_git_oid(context[name], name)
    for name in ("model_config_sha", "admission_manifest_sha", "outer_fold_sha", "selected_row_identity_sha",
                 "training_row_identity_sha", "operation_metadata_sha", "row_content_sha"):
        require_sha256(context[name], name)
    for name in ("sample_id", "group_id"):
        if type(context[name]) is not str or not context[name]:
            raise ValueError("Native state sample/group identifiers must be nonempty")
    if type(context["population_row_count"]) is not int or context["population_row_count"] < 1 \
            or type(context["session_index"]) is not int or context["session_index"] < 0 \
            or type(context["native_read_count"]) is not int or not 0 <= context["native_read_count"] <= 2**53 \
            or type(context["prepared_prior_available"]) is not bool:
        raise ValueError("Native state coverage/session/READ fields differ")
    recipe = context["query_recipe"]
    if type(recipe) is not dict or set(recipe) != set(asdict(Effective_inference_config())):
        raise ValueError("Native state query recipe fields differ")
    Effective_inference_config(**recipe).validate()
    operations = context["operation_ids"]
    if type(operations) is not list or not operations or len(set(operations)) != len(operations) \
            or any(type(value) is not str or not value for value in operations):
        raise ValueError("Native state operations must be distinct nonempty identifiers")
    size = len(operations)
    for name in view_fields[1:]:
        values = context[name]
        if type(values) is not list or len(values) != size:
            raise ValueError("Native state operation view shape differs")
    if any(type(value) is not str or not value for value in context["operation_families"]) \
            or any(type(value) is not int for value in context["cycle_indices"]) \
            or any(type(value) is not bool for name in view_fields[3:] for value in context[name]):
        raise ValueError("Native state operation view has invalid types")
    available, visible, query = (context[name] for name in view_fields[3:])
    if not any(visible) or not any(query) or any(q and not a or v != (a and not q)
            for a, v, q in zip(available, visible, query, strict=True)):
        raise ValueError("Native state visible/query masks differ")
    _validate_psd(context, "prepared_prior", 3 if context["system_type"] == "3qubit" else 4)


def select_native_state_row(payload: dict[str, Any], sample_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Select a row from an already validated point payload by identity."""
    if type(sample_id) is not str or not sample_id:
        raise ValueError("Native state selection requires an explicit sample ID")
    rows = [row for row in payload["rows"] if row["sample_id"] == sample_id]
    if len(rows) != 1:
        raise ValueError("Native state sample must occur exactly once in every point artifact")
    row = rows[0]
    protocol, model = payload["inference_protocol"], payload["model_artifact"]
    record = next(record for record in protocol["row_identity_records"] if record[0] == sample_id)
    session = dict(protocol["session_index_bindings"])[sample_id]
    context = {
        "schema": state_context_schema, "system_type": payload["system_type"], "model_schema": payload["model_schema"],
        "model_config_sha": model["model_config_sha"], "model_kind": model["kind"],
        "point_reader_commit": payload["source_commit"], "point_reader_tree": payload["source_tree"],
        "model_producer_commit": model["producer_commit"], "model_producer_tree": model["producer_tree"],
        "training_source_commit": model["training_source_commit"], "training_source_tree": model["training_source_tree"],
        **{name: protocol[name] for name in ("admission_manifest_sha", "outer_fold_sha", "outer_role",
            "selected_row_identity_sha", "training_row_identity_sha", "operation_metadata_sha", "query_recipe")},
        "population_row_count": protocol["row_count"], "sample_id": sample_id, "group_id": record[1],
        "row_content_sha": record[2], "source_role": record[3], "session_index": session,
        **{name: row[name] for name in view_fields},
        **{name: row["values"][name] for name in ("native_read_count", "prepared_prior_available",
            "prepared_prior_real", "prepared_prior_imag")},
    }
    validate_native_state_context(context)
    binding = protocol["batches"][row["batch_index"]]
    provenance = {
        "point_protocol_sha": payload["inference_protocol_sha"], "point_row_sha": canonical_sha256(row),
        "point_row_view_sha": canonical_sha256({name: row[name] for name in point_row_fields - {"values"}}),
        "batch_index": row["batch_index"], "batch_input_sha": row["input_sha"],
        "batch_query_mask_sha": binding["query_mask_sha"],
    }
    return context, row, provenance


def describe_native_state_context(artifact: Effective_point_inference, sample_id: str) -> dict[str, Any]:
    """Describe the expected state key without creating a model or prediction."""
    if type(artifact) is not Effective_point_inference:
        raise TypeError("Native state context requires an exact point artifact")
    context, _, _ = select_native_state_row(artifact.payload(), sample_id)
    return {"context": context, "state_key_sha": canonical_sha256(context)}
