"""Auditable real-observation posterior approximation for Track B research use."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from .contracts import canonical_sha256, require_sha256


def _finite_vector(values: Sequence[object], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must be nonempty and finite")
    return result


@dataclass(frozen=True)
class TrackBPosteriorV1:
    """Posterior/ensemble moments, not a claim of calibrated experimental truth."""

    system_type: str
    state_coordinate_mean: tuple[float, ...]
    state_coordinate_covariance: tuple[tuple[float, ...], ...]
    nuisance_mean: tuple[float, ...]
    nuisance_covariance: tuple[tuple[float, ...], ...]
    background_mean: float
    readout_efficiency_mean: float
    measurement_rf_residual_mean: tuple[float, ...]
    session_latent_mean: tuple[float, ...]
    ood_probability: float
    apparatus_confidence: float
    forward_sha: str
    covariance_sha: str
    forward_draw_set_sha: str | None = None
    forward_draw_ids_sha: str | None = None
    schema: str = field(default="TrackBPosteriorV1", init=False)

    def __post_init__(self) -> None:
        if self.system_type not in ("3qubit", "4qubit"):
            raise ValueError("posterior system type is unsupported")
        state = _finite_vector(self.state_coordinate_mean, "state_coordinate_mean")
        nuisance = _finite_vector(self.nuisance_mean, "nuisance_mean")
        residual = _finite_vector(self.measurement_rf_residual_mean, "measurement_rf_residual_mean")
        session = _finite_vector(self.session_latent_mean, "session_latent_mean")
        for matrix, expected, name in (
            (self.state_coordinate_covariance, len(state), "state_coordinate_covariance"),
            (self.nuisance_covariance, len(nuisance), "nuisance_covariance"),
        ):
            array = np.asarray(matrix, dtype=np.float64)
            if array.shape != (expected, expected) or not np.all(np.isfinite(array)) or not np.allclose(array, array.T, atol=1.0e-10):
                raise ValueError(f"{name} must be finite symmetric and align to its mean")
            if np.linalg.eigvalsh(array).min(initial=0.0) < -1.0e-10:
                raise ValueError(f"{name} must be positive semidefinite")
        if not all(math.isfinite(float(value)) for value in (
            self.background_mean, self.readout_efficiency_mean, self.ood_probability, self.apparatus_confidence,
        )):
            raise ValueError("posterior scalar fields must be finite")
        if self.background_mean < 0.0 or self.readout_efficiency_mean < 0.0:
            raise ValueError("background/readout quantities cannot be negative")
        if not 0.0 <= self.ood_probability <= 1.0 or not 0.0 <= self.apparatus_confidence <= 1.0:
            raise ValueError("OOD/confidence must lie in [0,1]")
        require_sha256(self.forward_sha, "forward_sha")
        require_sha256(self.covariance_sha, "covariance_sha")
        if self.forward_draw_set_sha is not None:
            require_sha256(self.forward_draw_set_sha, "forward_draw_set_sha")
        if self.forward_draw_ids_sha is not None:
            require_sha256(self.forward_draw_ids_sha, "forward_draw_ids_sha")
        if (self.forward_draw_set_sha is None) != (self.forward_draw_ids_sha is None):
            raise ValueError("posterior forward draw-set and draw-ID identities must arrive together")

    @property
    def uncertainty_scalar(self) -> float:
        return float(np.trace(np.asarray(self.state_coordinate_covariance, dtype=np.float64)))

    def to_dict(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema": self.schema,
            "system_type": self.system_type,
            "state_coordinate_mean": list(self.state_coordinate_mean),
            "state_coordinate_covariance": [list(row) for row in self.state_coordinate_covariance],
            "nuisance_mean": list(self.nuisance_mean),
            "nuisance_covariance": [list(row) for row in self.nuisance_covariance],
            "background_mean": self.background_mean,
            "readout_efficiency_mean": self.readout_efficiency_mean,
            "measurement_rf_residual_mean": list(self.measurement_rf_residual_mean),
            "session_latent_mean": list(self.session_latent_mean),
            "ood_probability": self.ood_probability,
            "apparatus_confidence": self.apparatus_confidence,
            "forward_sha": self.forward_sha,
            "covariance_sha": self.covariance_sha,
            "forward_draw_set_sha": self.forward_draw_set_sha,
            "forward_draw_ids_sha": self.forward_draw_ids_sha,
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


__all__ = ["TrackBPosteriorV1"]
