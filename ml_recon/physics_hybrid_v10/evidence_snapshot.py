"""Bounded evidence snapshots: hash and parse the same immutable byte buffer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .evidence_file_io import Evidence_file_identity, read_evidence_file


DEFAULT_JSON_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class Evidence_snapshot:
    """A local byte snapshot, not an issuer assertion or execution authority."""

    path: Path
    raw_bytes: bytes
    sha256: str
    file_identity: Evidence_file_identity


def strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate admission or review JSON key")
        result[key] = value
    return result


def reject_json_constant(value: str) -> None:
    raise ValueError(f"nonfinite admission or review JSON constant: {value}")


def read_snapshot(
    path: str | Path,
    name: str,
    *,
    expected_sha: str | None = None,
    expected_identity: Evidence_file_identity | None = None,
    maximum_bytes: int = DEFAULT_JSON_BYTES,
) -> Evidence_snapshot:
    """Open once, bound allocation, and bind the bytes before semantic use."""
    if type(maximum_bytes) is not int or maximum_bytes < 1:
        raise ValueError("evidence size limit must be a positive integer")
    if expected_sha is not None and (
        not isinstance(expected_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
    ):
        raise ValueError(f"{name} requires a canonical SHA-256")
    canonical_path, raw, file_identity = read_evidence_file(
        path,
        name,
        maximum_bytes,
        expected_identity=expected_identity,
    )
    observed = hashlib.sha256(raw).hexdigest()
    if expected_sha is not None and observed != expected_sha:
        raise ValueError(f"{name} bytes changed; snapshot SHA mismatch")
    return Evidence_snapshot(canonical_path, raw, observed, file_identity)


def parse_json_object(snapshot: Evidence_snapshot, name: str) -> dict[str, Any]:
    """Decode only the already-read bytes, with no path reopening."""
    value = parse_json_value(snapshot, name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def parse_json_value(snapshot: Evidence_snapshot, name: str) -> Any:
    """Decode any JSON value from one bounded, immutable byte snapshot."""
    try:
        value = json.loads(
            snapshot.raw_bytes.decode("utf-8"),
            object_pairs_hook=strict_object_pairs,
            parse_constant=reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be UTF-8 JSON") from exc
    return value


def load_json_snapshot(
    path: str | Path,
    name: str,
    *,
    expected_sha: str | None = None,
    expected_identity: Evidence_file_identity | None = None,
    maximum_bytes: int = DEFAULT_JSON_BYTES,
) -> tuple[Evidence_snapshot, dict[str, Any]]:
    snapshot = read_snapshot(
        path,
        name,
        expected_sha=expected_sha,
        expected_identity=expected_identity,
        maximum_bytes=maximum_bytes,
    )
    return snapshot, parse_json_object(snapshot, name)
