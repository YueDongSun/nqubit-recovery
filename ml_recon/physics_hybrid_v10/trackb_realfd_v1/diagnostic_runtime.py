"""Shared state-preservation helpers for Track B read-only diagnostics."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import random

import numpy as np
import torch


@contextmanager
def preserve_python_numpy_torch_rng() -> Iterator[None]:
    """Restore caller RNG states on success and on every propagated failure."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


__all__ = ["preserve_python_numpy_torch_rng"]
