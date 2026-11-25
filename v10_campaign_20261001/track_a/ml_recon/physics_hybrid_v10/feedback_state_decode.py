"""Versioned decoding for observation-derived feedback state estimates."""

from __future__ import annotations

from typing import Final

import numpy as np


LEGACY_ANALYTIC_STATE_DECODE: Final[str] = "legacy_phase_proxy_v1"
PHYSICAL_ANALYTIC_STATE_DECODE: Final[str] = "physical_conjugate_v2"
ANALYTIC_STATE_DECODE_VERSIONS: Final[tuple[str, str]] = (
    LEGACY_ANALYTIC_STATE_DECODE,
    PHYSICAL_ANALYTIC_STATE_DECODE,
)


def validate_analytic_state_decode_version(version: str) -> str:
    """Return a registered observation-state decode version."""
    if not isinstance(version, str) or version not in ANALYTIC_STATE_DECODE_VERSIONS:
        raise ValueError(f"unsupported analytic state decode version: {version!r}")
    return version


def decode_analytic_state(
    observation_tokens: np.ndarray,
    *,
    version: str = LEGACY_ANALYTIC_STATE_DECODE,
) -> np.ndarray:
    """Decode population and phase-proxy tokens using an explicit convention.

    ``legacy_phase_proxy_v1`` preserves the original feedback estimate exactly.
    ``physical_conjugate_v2`` converts that negative-phase observation proxy to
    the physical controller estimate by complex conjugation.
    """
    decode_version = validate_analytic_state_decode_version(version)
    tokens = np.asarray(observation_tokens)
    if (tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] != 2
            or tokens.dtype.kind not in ("i", "u", "f")
            or not np.all(np.isfinite(tokens))):
        raise ValueError("observation tokens must be a nonempty Nx2 finite real array")

    # Keep this operation order and dtype behavior aligned with the historical
    # estimate used before the decoder became versioned.
    population = np.maximum(tokens[:, 0], 1e-8)
    population /= population.sum()
    estimated = np.sqrt(population).astype(np.complex128)
    estimated *= np.exp(1j * np.pi * tokens[:, 1])
    if decode_version == PHYSICAL_ANALYTIC_STATE_DECODE:
        estimated = np.conjugate(estimated)
    return estimated
