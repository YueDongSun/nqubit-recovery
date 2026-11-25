"""Fail-closed replay checks for retired V2.4 synthetic-stream evidence.

The retired tensor corpus is deliberately never reopened.  This module reads
only retained READY metadata and a frozen identity/hash fixture, regenerates a
bounded set of rows in memory, and records whether that limited evidence is
sufficient for admission.  It is not a training, data-materialization, or
scientific-gate interface.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .hashing import read_canonical_json, sha256_file, sha256_json
from .rung_plan import plan_registered_rung
from .streaming import V24_stream_dataset, V24_stream_spec, classifier_stream_datasets


HISTORICAL_METADATA_SHA256 = "1bf9d59fb675515fa500f636811c05c0e9e28a836866e907ddbb07f51fbcdce3"
HISTORICAL_READY_PROJECTION_SHA256 = "97eefb25a17e8e3e535cde7f3e5ca97caa51aaff2636ab5d9cb7907fa4767b6f"
HISTORICAL_READY_MANIFEST_IDENTITIES_SHA256 = "c3d350b492f36b209f73328494724c6bcd2e048fca01d24a417664d445be8263"
HISTORICAL_AUDIT_FIXTURE_SHA256 = "ea123307fc300c87baeb2b143e266adcca9a0d931fad2f29066e98abd483ce10"
HISTORICAL_READY_MANIFEST_COUNT = 40
HISTORICAL_TOTAL_ROWS = 1840
HISTORICAL_AUDIT_ROWS = 120
HISTORICAL_TRAIN_ROWS = 1000
HISTORICAL_VALIDATION_ROWS = 200
RECEIPT_SCHEMA = "V24HistoricalStreamParityReceiptV1"

_PROJECTION_KEYS = frozenset({"manifests", "original_ready_metadata_sha256", "projection_provenance", "schema"})
_PROJECTION_MANIFEST_KEYS = frozenset({"audit_row_hashes", "build_id", "cell_id", "cell_static_file_sha", "cell_static_sha", "contract_sha", "generator_sha", "observation_sha", "original_ready_manifest_bytes", "original_ready_manifest_sha256", "row_count", "schema_sha", "shard_id", "split", "start_index", "stop_index", "truth_sha"})
_AUDIT_ROW_KEYS = frozenset({"cell", "index", "sample", "sha", "split"})


def _require_exact_keys(value: Mapping[str, Any], keys: frozenset[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} has unknown, missing, or noncanonical keys")


def _fixture_path() -> Path:
    return Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "v24_historical_audit_rows.json"


def _load_historical_metadata(metadata_path: str | Path) -> dict[str, Any]:
    """Load the public projection, never the private retained collector record."""

    path = Path(metadata_path)
    if sha256_file(path) != HISTORICAL_READY_PROJECTION_SHA256:
        raise ValueError("sanitized READY projection SHA-256 does not match the frozen expectation")
    value = read_canonical_json(path)
    if not isinstance(value, dict):
        raise ValueError("sanitized READY projection must be an object")
    _require_exact_keys(value, _PROJECTION_KEYS, "sanitized READY projection")
    if (value["schema"] != "V24HistoricalReadyProjectionV1"
            or value["original_ready_metadata_sha256"] != HISTORICAL_METADATA_SHA256
            or not isinstance(value["projection_provenance"], str)
            or not value["projection_provenance"].startswith("Deterministic public projection")):
        raise ValueError("sanitized READY projection provenance is invalid")
    manifests = value["manifests"]
    if not isinstance(manifests, list) or len(manifests) != HISTORICAL_READY_MANIFEST_COUNT:
        raise ValueError("sanitized READY manifest count is incomplete")
    for manifest in manifests:
        if not isinstance(manifest, dict):
            raise ValueError("sanitized READY manifest is not an object")
        _require_exact_keys(manifest, _PROJECTION_MANIFEST_KEYS, "sanitized READY manifest")
        if (not isinstance(manifest["original_ready_manifest_sha256"], str)
                or len(manifest["original_ready_manifest_sha256"]) != 64
                or type(manifest["original_ready_manifest_bytes"]) is not int
                or manifest["original_ready_manifest_bytes"] <= 0):
            raise ValueError("sanitized READY original-manifest identity is invalid")
    identities = [{"bytes": manifest["original_ready_manifest_bytes"],
                   "sha256": manifest["original_ready_manifest_sha256"]}
                  for manifest in manifests]
    if (len({item["sha256"] for item in identities}) != HISTORICAL_READY_MANIFEST_COUNT
            or sha256_json(identities) != HISTORICAL_READY_MANIFEST_IDENTITIES_SHA256):
        raise ValueError("sanitized READY per-manifest identity ledger is invalid")
    return value


def _load_expected_audit_rows(fixture_path: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(fixture_path) if fixture_path is not None else _fixture_path()
    if sha256_file(path) != HISTORICAL_AUDIT_FIXTURE_SHA256:
        raise ValueError("historical audit fixture SHA-256 does not match the frozen expectation")
    import json
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or len(rows) != HISTORICAL_AUDIT_ROWS:
        raise ValueError("historical audit fixture coverage is incomplete")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("historical audit row is not an object")
        _require_exact_keys(row, _AUDIT_ROW_KEYS, "historical audit row")
        if type(row["index"]) is not int or row["index"] < 0:
            raise ValueError("historical audit index is invalid")
        if not all(isinstance(row[key], str) and row[key] for key in ("cell", "split", "sample", "sha")):
            raise ValueError("historical audit row identity is invalid")
    return rows


def _metadata_row_identity(metadata: Mapping[str, Any]) -> set[tuple[str, str, int, str]]:
    identities: set[tuple[str, str, int, str]] = set()
    for manifest in metadata["manifests"]:
        for index, digest in manifest["audit_row_hashes"]:
            identities.add((manifest["cell_id"], manifest["split"], int(index), str(digest)))
    return identities


def _historical_coverage(metadata: Mapping[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for manifest in metadata["manifests"]:
        counts[manifest["split"]] += int(manifest["row_count"])
    if sum(counts.values()) != HISTORICAL_TOTAL_ROWS:
        raise ValueError("retained READY row total differs from the historical 1,840-row corpus")
    if counts["train"] != HISTORICAL_TRAIN_ROWS or counts["validation"] != HISTORICAL_VALIDATION_ROWS:
        raise ValueError("retained READY train/validation coverage differs from the frozen pilot contract")
    return dict(sorted(counts.items()))


def _historical_dataset_family(cell_id: str) -> str:
    if "_negative_control_" in cell_id:
        return "small_n_family_negative_control_" + cell_id.split("_negative_control_", 1)[1]
    if cell_id.endswith("_locked_ood"):
        return "small_n_family_locked_ood"
    return "small_n_family"


def replay_historical_audit_rows(metadata_path: str | Path, *, fixture_path: str | Path | None = None) -> dict[str, Any]:
    """Regenerate exactly the independently retained 120 hash probes in memory."""

    metadata = _load_historical_metadata(metadata_path)
    expected_rows = _load_expected_audit_rows(fixture_path)
    metadata_rows = _metadata_row_identity(metadata)
    if {(row["cell"], row["split"], row["index"], row["sha"]) for row in expected_rows} != metadata_rows:
        raise ValueError("frozen audit fixture and retained READY metadata disagree")
    plan = plan_registered_rung("small_n_classifier", HISTORICAL_TRAIN_ROWS)
    cells = {cell.cell_id: cell for cell in plan.cell_allocations}
    mismatches: list[dict[str, str]] = []
    for expected in expected_rows:
        cell = cells.get(expected["cell"])
        if cell is None:
            base_cell_id = expected["cell"].split("_audit", 1)[0].split("_locked_ood", 1)[0].split("_negative_control", 1)[0]
            cell = cells.get(base_cell_id)
            if cell is not None:
                cell = replace(cell, cell_id=expected["cell"],
                               dataset_family=_historical_dataset_family(expected["cell"]))
        if cell is None:
            raise ValueError("historical audit references an unregistered cell")
        dataset = V24_stream_dataset(V24_stream_spec(cell, expected["split"], sample_count=1, start_index=expected["index"]))
        actual = dataset[0]
        if actual["sample_id"] != expected["sample"] or actual["rendered_row_sha256"] != expected["sha"]:
            mismatches.append({"sample_id": expected["sample"], "expected_row_sha256": expected["sha"], "actual_row_sha256": actual["rendered_row_sha256"]})
    coverage = _historical_coverage(metadata)
    return {
        "schema": RECEIPT_SCHEMA,
        "historical_metadata_sha256": HISTORICAL_METADATA_SHA256,
        "historical_audit_fixture_sha256": HISTORICAL_AUDIT_FIXTURE_SHA256,
        "ready_manifest_count": len(metadata["manifests"]),
        "historical_rows": HISTORICAL_TOTAL_ROWS,
        "role_rows": coverage,
        "audit_expected_rows": len(expected_rows),
        "audit_checked_rows": len(expected_rows),
        "audit_mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "row_hash_coverage_complete": False,
        "admission_ready": False,
        "raw_sample_io": False,
        "optimizer_steps": 0,
        "scientific_gate_evaluated": False,
        "boundary": "120 retained row-hash probes match stream regeneration; the retired corpus has no retained 1,840-row hash ledger, so pilot admission remains blocked.",
    }


def validate_stream_historical_parity(manifest_path: str | Path, expected_sha256: str, *, recipe_payload_sha256: str, recipe_file_sha256: str, rung_plan_sha256: str) -> dict[str, object]:
    """Validate a canonical parity receipt for fail-closed pilot admission.

    A caller must require ``admission_ready``; this historical evidence currently
    always returns false because it preserves only 120 of 1,840 row digests.
    """

    if any(not isinstance(value, str) or len(value) != 64
           or any(character not in "0123456789abcdef" for character in value)
           for value in (expected_sha256, recipe_payload_sha256, recipe_file_sha256, rung_plan_sha256)):
        raise ValueError("all parity and recipe hashes must be lowercase SHA-256 strings")
    path = Path(manifest_path)
    if sha256_file(path) != expected_sha256:
        raise ValueError("parity receipt SHA-256 mismatch")
    receipt = read_canonical_json(path)
    if not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("parity receipt schema is invalid")
    expected_keys = {"schema", "historical_metadata_sha256", "historical_audit_fixture_sha256", "ready_manifest_count", "historical_rows", "role_rows", "audit_expected_rows", "audit_checked_rows", "audit_mismatch_count", "mismatches", "row_hash_coverage_complete", "admission_ready", "raw_sample_io", "optimizer_steps", "scientific_gate_evaluated", "boundary", "recipe_payload_sha256", "recipe_file_sha256", "rung_plan_sha256"}
    _require_exact_keys(receipt, frozenset(expected_keys), "parity receipt")
    if receipt["recipe_payload_sha256"] != recipe_payload_sha256 or receipt["recipe_file_sha256"] != recipe_file_sha256 or receipt["rung_plan_sha256"] != rung_plan_sha256:
        raise ValueError("parity receipt recipe or rung identity mismatch")
    plan = plan_registered_rung(
        "small_n_classifier", HISTORICAL_TRAIN_ROWS, HISTORICAL_VALIDATION_ROWS)
    _, _, recipe = classifier_stream_datasets(plan)
    if (rung_plan_sha256 != plan.sha256()
            or recipe_payload_sha256 != recipe["recipe_sha256"]):
        raise ValueError("parity receipt does not bind the current 1,000/200 stream recipe")
    if (receipt["historical_metadata_sha256"] != HISTORICAL_METADATA_SHA256
            or receipt["historical_audit_fixture_sha256"] != HISTORICAL_AUDIT_FIXTURE_SHA256
            or receipt["ready_manifest_count"] != HISTORICAL_READY_MANIFEST_COUNT
            or receipt["historical_rows"] != HISTORICAL_TOTAL_ROWS
            or receipt["role_rows"].get("train") != HISTORICAL_TRAIN_ROWS
            or receipt["role_rows"].get("validation") != HISTORICAL_VALIDATION_ROWS
            or set(receipt["role_rows"]) != {"audit", "locked_ood", "negative_control", "test", "train", "validation"}
            or receipt["role_rows"].get("audit") != 200
            or receipt["role_rows"].get("locked_ood") != 40
            or receipt["role_rows"].get("negative_control") != 200
            or receipt["role_rows"].get("test") != 200
            or receipt["audit_expected_rows"] != HISTORICAL_AUDIT_ROWS
            or receipt["audit_checked_rows"] != HISTORICAL_AUDIT_ROWS
            or receipt["audit_mismatch_count"] != 0
            or receipt["mismatches"] != []
            or receipt["row_hash_coverage_complete"] is not False
            or receipt["admission_ready"] is not False
            or receipt["raw_sample_io"] is not False
            or receipt["optimizer_steps"] != 0
            or receipt["scientific_gate_evaluated"] is not False):
        raise ValueError("parity receipt is incomplete or does not match the frozen historical evidence")
    return {"admission_ready": False, "reason": "historical row-hash coverage is 120 of 1,840; a complete independently retained ledger is required", "audit_checked_rows": receipt["audit_checked_rows"], "historical_rows": receipt["historical_rows"]}
