"""Read-only RNG format checks using private CPU generators, never global restore."""

from __future__ import annotations

import math
import random
from typing import Any, Mapping

import numpy as np
import torch


MT_WORD_COUNT = 624
UINT32_MAX = 2**32 - 1


def _byte_vector(value: Any, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu" \
            or value.dtype != torch.uint8 or value.ndim != 1 or value.numel() == 0 \
            or not value.is_contiguous():
        raise ValueError(f"{name} must be a nonempty contiguous CPU uint8 vector")


def validate_checkpoint_rng_structure(state: Mapping[str, Any]) -> None:
    """Check registered Python/NumPy/CPU Torch formats without altering caller RNG.

Saved CUDA states are checked only as CPU-mapped byte vectors. Their generator
semantics are intentionally not decoded or restored, and no CUDA API is used.
CPU Torch layout is validated by a local generator rather than a hardcoded
byte length that would incorrectly assume all Torch releases share a layout.
"""
    if not isinstance(state, Mapping) or set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("checkpoint RNG field set is invalid")
    python_state = state["python"]
    if not isinstance(python_state, (list, tuple)) or len(python_state) != 3 \
            or type(python_state[0]) is not int or python_state[0] != 3:
        raise ValueError("Python RNG requires the registered version-3 state")
    words = python_state[1]
    if not isinstance(words, (list, tuple)) or len(words) != MT_WORD_COUNT + 1 \
            or any(type(value) is not int or not 0 <= value <= UINT32_MAX for value in words[:-1]) \
            or type(words[-1]) is not int or not 0 <= words[-1] <= MT_WORD_COUNT:
        raise ValueError("Python RNG word vector or index is invalid")
    gaussian = python_state[2]
    if gaussian is not None and (type(gaussian) not in (int, float) or not math.isfinite(gaussian)):
        raise ValueError("Python RNG Gaussian cache is invalid")
    local_python = random.Random(0)
    local_python.setstate((3, tuple(words), gaussian))
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping) or set(numpy_state) != {
        "bit_generator", "state", "position", "has_gauss", "cached_gaussian",
    } or numpy_state["bit_generator"] != "MT19937":
        raise ValueError("NumPy RNG requires the registered MT19937 state")
    keys = numpy_state["state"]
    if not isinstance(keys, list) or len(keys) != MT_WORD_COUNT \
            or any(type(value) is not int or not 0 <= value <= UINT32_MAX for value in keys) \
            or type(numpy_state["position"]) is not int or not 0 <= numpy_state["position"] <= MT_WORD_COUNT \
            or type(numpy_state["has_gauss"]) is not int or numpy_state["has_gauss"] not in (0, 1) \
            or type(numpy_state["cached_gaussian"]) not in (int, float) \
            or not math.isfinite(numpy_state["cached_gaussian"]):
        raise ValueError("NumPy RNG vector, position or cache is invalid")
    local_numpy = np.random.RandomState(0)
    local_numpy.set_state(("MT19937", np.asarray(keys, dtype=np.uint32), numpy_state["position"],
                           numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
    _byte_vector(state["torch_cpu"], "Torch CPU RNG")
    local_torch = torch.Generator(device="cpu")
    try:
        local_torch.set_state(state["torch_cpu"])
    except (RuntimeError, TypeError) as error:
        raise ValueError("Torch CPU RNG layout is unsupported or invalid") from error
    if not isinstance(state["torch_cuda"], list):
        raise ValueError("saved CUDA RNG states must be a list")
    for value in state["torch_cuda"]:
        _byte_vector(value, "saved CUDA RNG")
