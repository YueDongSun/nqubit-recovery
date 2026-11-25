"""Isolated producer receipt emitter for Track B actionful posterior artifacts.

This module intentionally uses only the Python standard library so
``source_artifacts`` can execute an audited source-byte snapshot with
``python -I``.  It has no model, data loader, AWG, or hardware dependency.
Its sole responsibility is to reject an unregistered producer configuration
and issue a canonical receipt for the exact files it consumed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REGISTERED_POSTERIOR_PRODUCER_RELATIVE_PATH = (
    "ml_recon/physics_hybrid_v10/trackb_realfd_v1/posterior_producer.py"
)
REGISTERED_POSTERIOR_PRODUCER_QUALIFIED_NAME = "emit_trackb_posterior_execution_receipt"
REGISTERED_POSTERIOR_OUTPUT_SCHEMA = "TrackBPosteriorProducerOutputV1"
EXECUTION_RECEIPT_SCHEMA = "TrackBPosteriorProducerExecutionReceiptV2"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_canonical_json(path: Path, *, schema: str, fields: frozenset[str]) -> dict[str, object]:
    payload = path.read_bytes()

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("posterior producer JSON contains duplicate keys")
            result[key] = value
        return result

    try:
        decoded = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("posterior producer JSON is not canonical UTF-8") from error
    if type(decoded) is not dict or decoded.get("schema") != schema or set(decoded) != fields:
        raise ValueError("posterior producer JSON schema is invalid")
    if _canonical_json_bytes(decoded) != payload:
        raise ValueError("posterior producer JSON is not canonical")
    return decoded


def _require_git_oid(value: object, name: str) -> str:
    if type(value) is not str or len(value) not in (40, 64) \
            or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase Git object identity")
    return value


def emit_trackb_posterior_execution_receipt(request_path: str | Path) -> dict[str, object]:
    """Rehash registered producer inputs and emit their execution receipt."""
    request = _read_canonical_json(
        Path(request_path),
        schema="TrackBPosteriorProducerRequestV1",
        fields=frozenset({
            "schema", "source_commit", "source_tree", "producer_source_git_blob_oid",
            "artifact_paths",
        }),
    )
    artifact_paths = request["artifact_paths"]
    if type(artifact_paths) is not dict or not artifact_paths:
        raise ValueError("posterior producer request has no artifact paths")
    paths = {
        name: Path(value).resolve()
        for name, value in artifact_paths.items()
        if type(name) is str and type(value) is str
    }
    if len(paths) != len(artifact_paths) or any(not path.is_file() for path in paths.values()):
        raise ValueError("posterior producer request artifact path is invalid")
    required = {
        "posterior_payload", "model_checkpoint", "model_output_bundle", "data_manifest",
        "row_identity", "batch_plan", "producer_source", "producer_config",
        "calibration_coverage",
    }
    if not required.issubset(paths):
        raise ValueError("posterior producer request omits a required source artifact")
    config = _read_canonical_json(
        paths["producer_config"],
        schema="TrackBPosteriorProducerConfigV1",
        fields=frozenset({
            "schema", "producer_relative_path", "producer_qualified_name",
            "producer_recipe_sha", "output_schema",
        }),
    )
    if config["producer_relative_path"] != REGISTERED_POSTERIOR_PRODUCER_RELATIVE_PATH \
            or config["producer_qualified_name"] != REGISTERED_POSTERIOR_PRODUCER_QUALIFIED_NAME \
            or config["output_schema"] != REGISTERED_POSTERIOR_OUTPUT_SCHEMA:
        raise ValueError("posterior producer configuration is not registered")
    recipe_payload = {
        "schema": "TrackBPosteriorProducerRecipeV1",
        "producer_relative_path": config["producer_relative_path"],
        "producer_qualified_name": config["producer_qualified_name"],
        "output_schema": config["output_schema"],
    }
    if config["producer_recipe_sha"] != _sha256_bytes(_canonical_json_bytes(recipe_payload)):
        raise ValueError("posterior producer recipe identity is invalid")
    output = _read_canonical_json(
        paths["model_output_bundle"],
        schema=REGISTERED_POSTERIOR_OUTPUT_SCHEMA,
        fields=frozenset({
            "schema", "posterior_payload_sha", "model_checkpoint_sha", "data_manifest_sha",
            "row_identity_sha", "batch_plan_sha", "calibration_coverage_sha", "producer_config_sha",
        }),
    )
    for name, artifact_name in (
        ("posterior_payload_sha", "posterior_payload"),
        ("model_checkpoint_sha", "model_checkpoint"),
        ("data_manifest_sha", "data_manifest"),
        ("row_identity_sha", "row_identity"),
        ("batch_plan_sha", "batch_plan"),
        ("calibration_coverage_sha", "calibration_coverage"),
        ("producer_config_sha", "producer_config"),
    ):
        if output[name] != _sha256_bytes(paths[artifact_name].read_bytes()):
            raise ValueError("posterior producer output does not bind a consumed artifact")
    artifact_shas = {
        name: _sha256_bytes(path.read_bytes())
        for name, path in sorted(paths.items())
    }
    _require_git_oid(request["producer_source_git_blob_oid"], "producer_source_git_blob_oid")
    return {
        "schema": EXECUTION_RECEIPT_SCHEMA,
        "source_commit": request["source_commit"],
        "source_tree": request["source_tree"],
        "producer_relative_path": config["producer_relative_path"],
        "producer_qualified_name": config["producer_qualified_name"],
        "producer_recipe_sha": config["producer_recipe_sha"],
        "output_schema": config["output_schema"],
        "producer_source_git_blob_oid": request["producer_source_git_blob_oid"],
        "artifact_shas": artifact_shas,
    }


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        raise SystemExit("usage: posterior_producer.py <canonical-request-json>")
    receipt = emit_trackb_posterior_execution_receipt(arguments[0])
    sys.stdout.write(_canonical_json_bytes(receipt).decode("utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover - invoked from audited subprocess.
    raise SystemExit(main())
