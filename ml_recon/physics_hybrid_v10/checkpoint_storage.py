"""Detached CPU snapshots for portable, immutable training checkpoint storage."""
from __future__ import annotations

import copy
from typing import Any

import torch


def cpu_checkpoint_snapshot(value: Any) -> Any:
    """Do not pass live CUDA storages or borrowed model tensors to ZIP writing."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        result = copy.copy(value)
        for key, item in value.items():
            result[key] = cpu_checkpoint_snapshot(item)
        if hasattr(value, "_metadata"):
            result._metadata = copy.deepcopy(value._metadata)
        return result
    if isinstance(value, tuple):
        return tuple(cpu_checkpoint_snapshot(item) for item in value)
    if isinstance(value, list):
        return [cpu_checkpoint_snapshot(item) for item in value]
    return copy.deepcopy(value)
