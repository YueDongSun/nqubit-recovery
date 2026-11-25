"""Validate full source-to-stream equivalence without inventing historical rows.

This is an engineering migration contract. The retained 120 historical probes
remain historical evidence; the other rows are independently regenerated from a
frozen Git source and never described as recovered historical payloads.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


MIGRATION_SCHEMA = "V24StreamSourceMigrationV1"
WORKER_SCHEMA = "V24StreamMigrationWorkerV1"
REFERENCE_COMMIT = "65c000b4a229c2ecb199939d4b82b5c07277d197"
REFERENCE_TREE = "6af0ea8a27b48b6b62fab69d84489eb1bc83542e"
PROJECTION_SHA256 = "97eefb25a17e8e3e535cde7f3e5ca97caa51aaff2636ab5d9cb7907fa4767b6f"
GLOBAL_SEED = 20261001
TRAIN_ROWS = 1000
VALIDATION_ROWS = 200
TOTAL_ROWS = 1840
ANCHOR_ROWS = 120
SHARD_COUNT = 40
THREAD_COUNT = 2
MAX_EVIDENCE_BYTES = 4 * 1024 * 1024
ROW_FIELDS = frozenset({
    "cell_id", "split", "index", "sample_id", "rendered_row_sha256",
    "input_view_sha256", "truth_view_sha256",
})
WORKER_FIELDS = frozenset({
    "schema", "mode", "source", "worker_blobs", "runtime", "projection_sha256",
    "global_seed", "rows", "rows_completed", "terminal_status",
    "raw_payloads_persisted", "optimizer_steps",
})
WORKER_PATHS = (
    "scripts/v10_stream_migration_worker.py",
    "ml_recon/data_v2_4/stream_migration_contract.py",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def require_digest(value: object, label: str, length: int = 64) -> str:
    if (not isinstance(value, str) or len(value) != length
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{label} must be a lowercase digest of length {length}")
    return value


def load_projection(path: str | Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROJECTION_SHA256:
        raise ValueError("migration requires the frozen historical scope projection")
    return json.loads(raw)


def migration_scope(projection: Mapping[str, Any]) -> list[tuple[str, str, int]]:
    return [(item["cell_id"], item["split"], index)
            for item in projection["manifests"]
            for index in range(item["start_index"], item["stop_index"])]


def cell_family(cell_id: str) -> str:
    if "_negative_control_" in cell_id:
        return "small_n_family_negative_control_" + cell_id.split("_negative_control_", 1)[1]
    return "small_n_family_locked_ood" if cell_id.endswith("_locked_ood") else "small_n_family"


def expected_sample_id(cell_id: str, split: str, index: int) -> str:
    n = 3 if cell_id.startswith("classifier_n3_") else 4
    return f"v2_4:{cell_family(cell_id)}:{cell_id}:{split}:N{n}:{index:012d}"


def _validate_worker(
    report: Mapping[str, Any], *, mode: str, source: Mapping[str, str],
    worker_blobs: Mapping[str, str], runtime: Mapping[str, str],
    projection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(report, dict) or set(report) != WORKER_FIELDS:
        raise ValueError("migration worker report fields are invalid")
    expected = {
        "schema": WORKER_SCHEMA, "mode": mode, "source": dict(source),
        "worker_blobs": dict(worker_blobs), "runtime": dict(runtime),
        "projection_sha256": PROJECTION_SHA256, "global_seed": GLOBAL_SEED,
        "terminal_status": "completed", "rows_completed": TOTAL_ROWS,
    }
    if any(report[key] != value for key, value in expected.items()):
        raise ValueError("migration worker source, runtime, scope or terminal identity mismatch")
    if (report["raw_payloads_persisted"] is not False
            or type(report["optimizer_steps"]) is not int or report["optimizer_steps"] != 0
            or type(report["rows_completed"]) is not int
            or type(report["global_seed"]) is not int):
        raise ValueError("migration worker must preserve the no-training metadata-only boundary")
    rows = report["rows"]
    expected_scope = migration_scope(projection)
    if not isinstance(rows, list) or len(rows) != TOTAL_ROWS or len(expected_scope) != TOTAL_ROWS:
        raise ValueError("migration requires all 1840 rows, not a partial receipt")
    for row, key in zip(rows, expected_scope, strict=True):
        if (not isinstance(row, dict) or set(row) != ROW_FIELDS
                or type(row["index"]) is not int
                or (row["cell_id"], row["split"], row["index"]) != key):
            raise ValueError("migration row coverage/order mismatch")
        if row["sample_id"] != expected_sample_id(*key):
            raise ValueError("migration sample key differs from frozen membership")
        for field in ("rendered_row_sha256", "input_view_sha256", "truth_view_sha256"):
            require_digest(row[field], field)
    anchors = {(item["cell_id"], item["split"], index): value
               for item in projection["manifests"] for index, value in item["audit_row_hashes"]}
    if len(anchors) != ANCHOR_ROWS:
        raise ValueError("historical anchor coverage differs from the frozen projection")
    for row in rows:
        key = (row["cell_id"], row["split"], row["index"])
        if key in anchors and row["rendered_row_sha256"] != anchors[key]:
            raise ValueError("migration diverges from an independently retained historical anchor")
    return rows


def compare_reports(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], *,
    candidate_source: Mapping[str, str], worker_blobs: Mapping[str, str],
    runtime: Mapping[str, str], projection: Mapping[str, Any],
) -> None:
    reference_rows = _validate_worker(
        reference, mode="reference", source={"commit": REFERENCE_COMMIT, "tree": REFERENCE_TREE},
        worker_blobs=worker_blobs, runtime=runtime, projection=projection)
    candidate_rows = _validate_worker(
        candidate, mode="stream", source=candidate_source, worker_blobs=worker_blobs,
        runtime=runtime, projection=projection)
    if reference_rows != candidate_rows:
        raise ValueError("frozen-source and stream rows or named views differ")


def build_migration_evidence(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], *,
    candidate_source: Mapping[str, str], worker_blobs: Mapping[str, str],
    runtime: Mapping[str, str], projection: Mapping[str, Any],
    rung_plan_sha256: str, recipe_payload_sha256: str, recipe_file_sha256: str,
) -> dict[str, Any]:
    compare_reports(reference, candidate, candidate_source=candidate_source,
                    worker_blobs=worker_blobs, runtime=runtime, projection=projection)
    for value in (rung_plan_sha256, recipe_payload_sha256, recipe_file_sha256):
        require_digest(value, "recipe/plan identity")
    return {
        "schema": MIGRATION_SCHEMA,
        "reference": dict(reference), "candidate": dict(candidate),
        "rung_plan_sha256": rung_plan_sha256,
        "recipe_payload_sha256": recipe_payload_sha256,
        "recipe_file_sha256": recipe_file_sha256,
        "source_equivalence_verified": True,
        "historical_corpus_byte_parity_verified": False,
        "scientific_gate_evaluated": False,
    }


def validate_migration_evidence(
    path: str | Path, expected_sha256: str, *, candidate_source: Mapping[str, str],
    worker_blobs: Mapping[str, str], runtime: Mapping[str, str],
    projection_path: str | Path, rung_plan_sha256: str,
    recipe_payload_sha256: str, recipe_file_sha256: str,
) -> dict[str, Any]:
    require_digest(expected_sha256, "migration evidence")
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_EVIDENCE_BYTES + 1)
    if len(raw) > MAX_EVIDENCE_BYTES or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("migration evidence byte identity mismatch")
    report = json.loads(raw)
    if canonical_bytes(report) != raw or not isinstance(report, dict):
        raise ValueError("migration evidence must be a canonical JSON object")
    if (report.get("source_equivalence_verified") is not True
            or report.get("historical_corpus_byte_parity_verified") is not False
            or report.get("scientific_gate_evaluated") is not False):
        raise ValueError("migration evidence flags must be strict engineering-only booleans")
    rebuilt = build_migration_evidence(
        report.get("reference"), report.get("candidate"), candidate_source=candidate_source,
        worker_blobs=worker_blobs, runtime=runtime, projection=load_projection(projection_path),
        rung_plan_sha256=rung_plan_sha256, recipe_payload_sha256=recipe_payload_sha256,
        recipe_file_sha256=recipe_file_sha256)
    if report != rebuilt:
        raise ValueError("migration evidence declarations or recipe binding are invalid")
    return {"source_equivalence_verified": True, "historical_corpus_byte_parity_verified": False,
            "rows_checked": TOTAL_ROWS, "historical_anchors_checked": ANCHOR_ROWS}
