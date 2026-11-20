"""AOD-grid retrieval map inside the measurement basis.

The map is 16x16 for 4Q and the top-left 15x15 for 3Q. A measured idler
histogram initializes it. If that CSV is absent the prior is flat (all ones).
Each operation point's amplitude vector is multiplied, element by element, by
the retrieval rates of its cells before any operator is built. Count
re-weighting and clipping are not part of this path.

rate_power 1.0 keeps the historical rate scale. rate_power 0.5 treats the
rates as amplitude factors and is diagnostic only.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn

GRID = {
    "3qubit": {"rows": 15, "cols": 15, "group_size": 3, "groups_per_row": 5, "entries_per_cycle": 75},
    "4qubit": {"rows": 16, "cols": 16, "group_size": 4, "groups_per_row": 4, "entries_per_cycle": 64},
}
IDLER_ROW_LABEL = "Idler Absolute"
IDLER_COUNT = 256


def _spec(system_type: str) -> dict:
    if system_type not in GRID:
        raise ValueError(f"retrieval grid is defined for 3qubit and 4qubit, got {system_type!r}")
    return GRID[system_type]


def operation_cells(system_type: str, entry_index: int) -> tuple[int, list[int]]:
    """Map a basis entry to (row, columns) on the AOD grid.

    local = entry % entries_per_cycle
    row = local // groups_per_row
    group = local % groups_per_row
    columns = group * group_size .. + group_size
    4Q cycles reuse the same 16x16 map because of the modulo.
    """
    spec = _spec(system_type)
    if entry_index < 0:
        raise ValueError(f"entry index {entry_index} is negative")
    local = entry_index % spec["entries_per_cycle"]
    row = local // spec["groups_per_row"]
    group = local % spec["groups_per_row"]
    start = group * spec["group_size"]
    columns = list(range(start, start + spec["group_size"]))
    if row >= spec["rows"] or columns[-1] >= spec["cols"]:
        raise AssertionError(f"entry {entry_index} landed outside the {system_type} grid")
    return row, columns


def rate_vector(retrieval_map: np.ndarray, system_type: str, entry_index: int) -> np.ndarray:
    spec = _spec(system_type)
    grid = np.asarray(retrieval_map, dtype=np.float64)
    if grid.shape != (spec["rows"], spec["cols"]):
        raise ValueError(f"{system_type} retrieval map must have shape {(spec['rows'], spec['cols'])}, got {grid.shape}")
    row, columns = operation_cells(system_type, entry_index)
    return grid[row, columns].copy()


def scale_amplitudes(amplitudes, rates, rate_power: float) -> np.ndarray:
    """Elementwise amplitude * rate**rate_power. This is the operator input."""
    if rate_power <= 0:
        raise ValueError(f"rate_power must be positive, got {rate_power}")
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    rates = np.asarray(rates, dtype=np.float64)
    if amplitudes.shape != rates.shape:
        raise ValueError(f"amplitude length {amplitudes.shape} != rate length {rates.shape}")
    if np.any(rates < 0):
        raise ValueError("retrieval rates must be non-negative")
    return amplitudes * np.power(rates, rate_power)


def load_idler_histogram(path: str | Path, system_type: str) -> np.ndarray:
    """Row 'Idler Absolute', 256 counts, divide by the positive maximum, reshape 16x16.

    3Q keeps the top-left 15x15. 4Q keeps the full 16x16.
    """
    spec = _spec(system_type)
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    target = IDLER_ROW_LABEL.casefold()
    values = None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            if row[0].strip().casefold() == target:
                values = [float(item) for item in row[1:] if str(item).strip()]
                break
    if values is None:
        raise ValueError(f"CSV row {IDLER_ROW_LABEL!r} not found in {path}")
    counts = np.asarray(values, dtype=np.float64)
    if counts.size != IDLER_COUNT:
        raise ValueError(f"{IDLER_ROW_LABEL!r} must contain 256 counts, got {counts.size}")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("idler histogram has a negative or non-finite count")
    positive = counts[counts > 0]
    if positive.size == 0:
        raise ValueError("idler histogram has no positive count")
    matrix = (counts / float(positive.max())).reshape(16, 16)
    if system_type == "3qubit":
        return matrix[: spec["rows"], : spec["cols"]].copy()
    return matrix


def flat_prior(system_type: str) -> np.ndarray:
    spec = _spec(system_type)
    return np.ones((spec["rows"], spec["cols"]), dtype=np.float64)


def load_retrieval_prior(system_type: str, path: str | Path | None) -> np.ndarray:
    """Measured map when the CSV exists. Flat ones when the path is missing or empty."""
    if path is None or str(path).strip() == "":
        return flat_prior(system_type)
    file_path = Path(path)
    if not file_path.is_file():
        return flat_prior(system_type)
    return load_idler_histogram(file_path, system_type)


class LearnableRetrievalMap(nn.Module):
    """Per-cell map. delta = 0 reproduces the measured (or flat) prior.

    The weak prior is mean(delta^2). It pulls the map back toward that prior
    and is added to the likelihood loss by the caller. There is no count
    re-weighting and no clip to [0.1, 10].
    """

    def __init__(self, prior: np.ndarray):
        super().__init__()
        array = np.asarray(prior, dtype=np.float64)
        if array.ndim != 2 or np.any(array <= 0):
            raise ValueError("retrieval prior must be a positive 2D map")
        self.register_buffer("prior", torch.as_tensor(array, dtype=torch.float64))
        self.delta = nn.Parameter(torch.zeros_like(self.prior))

    def forward(self) -> torch.Tensor:
        eta = self.prior * torch.exp(self.delta)
        if not torch.isfinite(eta).all() or torch.any(eta <= 0):
            raise AssertionError("retrieval map left the positive reals")
        return eta

    def prior_penalty(self) -> torch.Tensor:
        return self.delta.square().mean()


def prior_from_config(system_type: str, config: dict) -> np.ndarray:
    if "idler_histogram_path" not in config:
        raise KeyError("missing config key idler_histogram_path")
    if "rate_power" not in config:
        raise KeyError("missing config key rate_power")
    return load_retrieval_prior(system_type, config["idler_histogram_path"])
