"""Independent NumPy reference for observable joint signal/idler outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
import math

import numpy as np

from ml_recon.physics_hybrid_v10.four_cell_layout import TRACK_A_FOUR_CELL_SEMANTICS

FOUR_CELL_SEMANTICS = TRACK_A_FOUR_CELL_SEMANTICS


@dataclass(frozen=True)
class Aod_noise_model:
    state_visibility: float = 1.0
    apparatus_efficiency: float = 1.0
    detector_dark_probability: float = 0.0
    per_cell_success_probability: float = 0.005

    def validate(self) -> None:
        for name in (
            "state_visibility",
            "apparatus_efficiency",
            "detector_dark_probability",
            "per_cell_success_probability",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be a finite scalar probability")


def four_cell_probabilities(
    *,
    ideal_projector_probability: float,
    throughput: float,
    noise: Aod_noise_model,
    herald_probability: float | None = None,
) -> np.ndarray:
    """Return [00,01,10,11] without exposing unobserved photon-source classes.

    State dephasing is applied before this detector boundary. Visibility is not
    multiplied into retrieval efficiency: doing so would replace decoherence by
    photon loss. The caller supplies the already noise-rendered projection.
    """
    noise.validate()
    signal = (
        noise.per_cell_success_probability
        if herald_probability is None
        else herald_probability
    )
    for name, value in (
        ("projection", ideal_projector_probability),
        ("throughput", throughput),
        ("herald", signal),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{name} must lie in [0,1]")
    dark = noise.detector_dark_probability
    conditional = 1.0 - (
        1.0 - noise.apparatus_efficiency * throughput * ideal_projector_probability
    ) * (1.0 - dark)
    values = np.asarray(
        [
            (1.0 - signal) * (1.0 - dark),
            (1.0 - signal) * dark,
            signal * (1.0 - conditional),
            signal * conditional,
        ],
        dtype=np.float64,
    )
    if np.any(values < 0) or not np.isclose(values.sum(), 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError("joint detector probabilities are not conserved")
    return values
