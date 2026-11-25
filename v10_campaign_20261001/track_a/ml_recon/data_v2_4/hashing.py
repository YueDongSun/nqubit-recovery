"""Canonical JSON, file, array, and tree hashing for V2.4 identities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = canonical_json_bytes({"dtype": array.dtype.str, "shape": list(array.shape)})
    return sha256_bytes(header + array.view(np.uint8).tobytes())


def combined_file_sha(paths: Iterable[str | Path], root: str | Path) -> str:
    root_path = Path(root).resolve()
    payload = []
    for path in sorted((Path(value).resolve() for value in paths), key=lambda p: p.as_posix()):
        payload.append({"path": path.relative_to(root_path).as_posix(), "sha256": sha256_file(path)})
    return sha256_json(payload)


def write_canonical_json(path: str | Path, value: Any) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        import os
        os.fsync(handle.fileno())
    temporary.replace(destination)
    return sha256_bytes(payload)


def read_canonical_json(path: str | Path) -> Any:
    payload = Path(path).read_bytes()
    value = json.loads(payload.decode("utf-8"))
    if payload != canonical_json_bytes(value):
        raise ValueError(f"JSON is not canonical: {path}")
    return value
