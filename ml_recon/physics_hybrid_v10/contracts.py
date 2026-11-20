"""Stable canonical identities for synthetic partitions."""
from typing import Any
import hashlib
import json


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":")).encode("utf-8")


def sha256_of(value: Any) -> str:
    data = value if isinstance(value, bytes) else stable_json_bytes(value)
    return hashlib.sha256(data).hexdigest()
