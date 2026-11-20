"""Detection efficiency chain and calibration maps (DECISIONS I1, J0, J1, J2).

Idler arm: intrinsic atom-to-first-optic retrieval 0.15, AOD 0.7, cavity 0.7,
detector 0.7 (product 0.0515). The PI's measured overall retrieval, 0.04, is the
absolute scale, so the chain is rescaled to a grid mean of 0.04. The idler
calibration histogram supplies only the relative per-cell variation.

Signal arm: one scalar eta_s = 0.2 (PI, 2026-11-03). A per-cell Stokes
efficiency is not separable from the write weight.

ASSUMPTION (J1): idler counts are proportional to detection probability, so the
per-cell efficiency is proportional to the map and the readout amplitude carries
sqrt(eta).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GRID_SIDE = 16
GRID_CELLS = GRID_SIDE * GRID_SIDE
THREE_QUBIT_SIDE = 15


@dataclass(frozen=True)
class EfficiencyChain:
    intrinsic: float
    aod: float
    cavity: float
    detector: float
    measured_mean_idler: float
    eta_signal: float

    @property
    def product(self) -> float:
        return self.intrinsic * self.aod * self.cavity * self.detector

    @property
    def scale(self) -> float:
        """Factor that maps the nominal chain product onto the measured mean."""
        return self.measured_mean_idler / self.product


def chain_from_config(cfg: dict) -> EfficiencyChain:
    chain = EfficiencyChain(
        intrinsic=float(cfg["intrinsic"]),
        aod=float(cfg["aod"]),
        cavity=float(cfg["cavity"]),
        detector=float(cfg["detector"]),
        measured_mean_idler=float(cfg["measured_mean_idler"]),
        eta_signal=float(cfg["eta_signal"]),
    )
    for name in ("intrinsic", "aod", "cavity", "detector", "measured_mean_idler", "eta_signal"):
        value = getattr(chain, name)
        if not 0.0 < value <= 1.0:
            raise ValueError(f"efficiency factor {name} = {value} is outside (0, 1]")
    return chain


def load_histogram_row(path: str | Path, label: str) -> np.ndarray:
    """One 256-count row of the calibration CSV as a raw 16x16 map (no normalization)."""
    path = Path(path)
    target = label.casefold()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if row and row[0].strip().casefold() == target:
                values = np.array([float(item) for item in row[1:] if str(item).strip()], dtype=np.float64)
                if values.size != GRID_CELLS:
                    raise ValueError(f"row {label!r} has {values.size} counts, expected {GRID_CELLS}")
                if np.any(values <= 0) or not np.all(np.isfinite(values)):
                    raise ValueError(f"row {label!r} has a non-positive or non-finite count")
                return values.reshape(GRID_SIDE, GRID_SIDE)
    raise ValueError(f"row {label!r} not found in {path}")


def system_window(raw_map: np.ndarray, system: str) -> np.ndarray:
    """3Q reads the top-left 15x15 window (C5); every other system uses the full grid."""
    if raw_map.shape != (GRID_SIDE, GRID_SIDE):
        raise ValueError(f"calibration map must be 16x16, got {raw_map.shape}")
    if system == "3qubit":
        return raw_map[:THREE_QUBIT_SIDE, :THREE_QUBIT_SIDE]
    return raw_map


def idler_efficiency_map(chain: EfficiencyChain, raw_idler: np.ndarray, system: str) -> np.ndarray:
    """Absolute per-cell idler efficiency on the full 16x16 grid.

    eta_j = measured_mean_idler * m_j / mean(m), with the mean over the system
    window. Cells outside a 3Q window keep the same scale factor; they are never read.
    """
    window = system_window(raw_idler, system)
    eta = chain.measured_mean_idler * raw_idler / window.mean()
    if eta.max() >= 1.0:
        raise AssertionError(f"per-cell idler efficiency {eta.max()} reached 1")
    return eta


def log_weight_pattern(raw_signal: np.ndarray, system: str) -> np.ndarray:
    """Signal-map log pattern for amplitude non-uniformity (J10), zero mean over the window."""
    window = system_window(raw_signal, system)
    log_map = np.log(raw_signal)
    return log_map - np.log(window).mean()
