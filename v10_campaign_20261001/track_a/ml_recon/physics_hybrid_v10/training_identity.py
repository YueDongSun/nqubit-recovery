"""Canonical identities shared by V10 training entrypoints and the core runtime."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from .contracts import sha256_of


def config_identity_sha(config: Any) -> str:
    """Return the execution-independent SHA for a V10 training config.

    ``resume_checkpoint`` selects transport for an execution and therefore is
    deliberately excluded.  Keeping this in a leaf module prevents the core
    trainer and the streaming entrypoint from developing different identity
    semantics.
    """
    payload = asdict(config) if is_dataclass(config) else dict(config)
    payload.pop("resume_checkpoint", None)
    return sha256_of(payload)
