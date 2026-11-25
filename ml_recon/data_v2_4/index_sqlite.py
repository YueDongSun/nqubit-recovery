"""SQLite index schema, integrity checks, and immutable row lookup."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .hashing import sha256_file


INDEX_COLUMNS = (
    "sample_id", "latent_state_id", "transition_id", "crn_group_id",
    "session_id", "acquisition_run_id", "repeat_group_id", "trajectory_id",
    "provenance", "dataset_family", "cell_id", "split", "system_type", "N",
    "measurement_domain", "observation_shard", "observation_row", "truth_shard",
    "truth_row", "sample_key_sha", "row_content_sha", "contract_sha", "build_id",
    "derived_rho8", "derived_count_mle", "derived_classifier", "derived_control",
)


def create_index(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(
            "CREATE TABLE samples ("
            "sample_id TEXT PRIMARY KEY, latent_state_id TEXT NOT NULL, transition_id TEXT NOT NULL, "
            "crn_group_id TEXT NOT NULL, session_id TEXT NOT NULL, acquisition_run_id TEXT NOT NULL, "
            "repeat_group_id TEXT NOT NULL, trajectory_id TEXT NOT NULL, provenance TEXT NOT NULL, "
            "dataset_family TEXT NOT NULL, cell_id TEXT NOT NULL, split TEXT NOT NULL, "
            "system_type TEXT NOT NULL, N INTEGER NOT NULL, measurement_domain TEXT NOT NULL, "
            "observation_shard TEXT NOT NULL, observation_row INTEGER NOT NULL, "
            "truth_shard TEXT NOT NULL, truth_row INTEGER NOT NULL, sample_key_sha TEXT NOT NULL, "
            "row_content_sha TEXT NOT NULL, contract_sha TEXT NOT NULL, build_id TEXT NOT NULL, "
            "derived_rho8 INTEGER NOT NULL, derived_count_mle INTEGER NOT NULL, "
            "derived_classifier INTEGER NOT NULL, derived_control INTEGER NOT NULL)")
        placeholders = ",".join("?" for _ in INDEX_COLUMNS)
        statement = f"INSERT INTO samples ({','.join(INDEX_COLUMNS)}) VALUES ({placeholders})"
        normalized = []
        for row in rows:
            missing = set(INDEX_COLUMNS) - set(row)
            if missing:
                raise ValueError(f"index row missing columns: {sorted(missing)}")
            normalized.append(tuple(row[column] for column in INDEX_COLUMNS))
        connection.executemany(statement, normalized)
        connection.execute("CREATE INDEX sample_cell_split ON samples(cell_id,split,sample_id)")
        connection.execute("CREATE INDEX sample_groups ON samples(session_id,trajectory_id,repeat_group_id)")
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")
        count = int(connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0])
        if count != len(normalized):
            raise RuntimeError("SQLite index row count mismatch")
    finally:
        connection.close()
    temporary.replace(destination)
    return {"path": str(destination), "rows": count, "integrity": "ok",
            "sha256": sha256_file(destination)}


def validate_index(path: str | Path, expected_sha: str | None = None) -> Dict[str, Any]:
    source = Path(path)
    actual_sha = sha256_file(source)
    if expected_sha is not None and actual_sha != expected_sha:
        raise ValueError("V2.4 index SHA mismatch")
    connection = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        count = int(connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0])
        duplicate = int(connection.execute(
            "SELECT COUNT(*) FROM (SELECT sample_id FROM samples GROUP BY sample_id HAVING COUNT(*)>1)"
        ).fetchone()[0])
    finally:
        connection.close()
    if integrity != "ok" or duplicate:
        raise ValueError("V2.4 SQLite index is corrupt or contains duplicate sample IDs")
    return {"sha256": actual_sha, "integrity": integrity, "rows": count}


def query_index(path: str | Path, *, cell_id: str, split: str) -> List[Dict[str, Any]]:
    source = Path(path)
    connection = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT * FROM samples WHERE cell_id=? AND split=? ORDER BY sample_id",
            (cell_id, split),
        ).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]


def all_index_rows(path: str | Path) -> List[Dict[str, Any]]:
    connection = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute("SELECT * FROM samples ORDER BY sample_id").fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]
