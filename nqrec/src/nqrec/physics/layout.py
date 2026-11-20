"""Physical cell layouts on the 16x16 AOD grid (DECISIONS H1, J12, J17).

Cells are indexed row-major, cell = 16 * row + column.

Hardware design (3Q, 4Q): operation i reads its own cell group (H1 map,
nqrec.physics.retrieval.operation_cells). The N logical modes of the W state are
the group's N cells, so every operation point uses different physical cells.

Fixed-cell design (any N): the N modes are N fixed cells, a row-major block of
width ceil(sqrt(N)) placed on the grid. N = 256 is the whole grid.
"""

from __future__ import annotations

import math

import numpy as np

from nqrec.physics.efficiency import GRID_SIDE
from nqrec.physics.retrieval import operation_cells

HARDWARE_OPERATIONS = {3: 75, 4: 256}
SYSTEM_NAME = {3: "3qubit", 4: "4qubit"}


def cell_xy(cells: np.ndarray) -> np.ndarray:
    cells = np.asarray(cells, dtype=np.int64)
    return np.stack([cells // GRID_SIDE, cells % GRID_SIDE], axis=-1).astype(np.float64)


def hardware_groups(n_modes: int) -> np.ndarray:
    """(operations, N) physical cell of logical mode q at operation i."""
    if n_modes not in HARDWARE_OPERATIONS:
        raise ValueError(f"the hardware design exists for N = 3 and 4, got {n_modes}")
    system = SYSTEM_NAME[n_modes]
    groups = np.zeros((HARDWARE_OPERATIONS[n_modes], n_modes), dtype=np.int64)
    for entry in range(HARDWARE_OPERATIONS[n_modes]):
        row, columns = operation_cells(system, entry)
        groups[entry] = [GRID_SIDE * row + column for column in columns]
    return groups


def block_shape(n_modes: int) -> tuple[int, int]:
    width = math.ceil(math.sqrt(n_modes))
    height = math.ceil(n_modes / width)
    return height, width


def fixed_cell_block(n_modes: int, rng: np.random.Generator) -> np.ndarray:
    """N cells, row-major in a block of width ceil(sqrt(N)), at a random valid offset."""
    if not 2 <= n_modes <= GRID_SIDE * GRID_SIDE:
        raise ValueError(f"fixed-cell design needs 2 <= N <= 256, got {n_modes}")
    height, width = block_shape(n_modes)
    row0 = int(rng.integers(0, GRID_SIDE - height + 1))
    col0 = int(rng.integers(0, GRID_SIDE - width + 1))
    local = np.arange(n_modes)
    rows = row0 + local // width
    cols = col0 + local % width
    return GRID_SIDE * rows + cols


def grid_distance(cells_a: np.ndarray, cells_b: np.ndarray) -> np.ndarray:
    """Euclidean distance in cell units between every pair."""
    xy_a = cell_xy(cells_a)
    xy_b = cell_xy(cells_b)
    diff = xy_a[:, None, :] - xy_b[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))
