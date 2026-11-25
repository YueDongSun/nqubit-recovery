"""Cooperative RAM and elapsed-time limits for research training processes.

Checks run at batch boundaries, not as a hard operating-system memory limit.
They include live child RSS so loader processes cannot escape the accounting.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict


class Resource_limit_exceeded(RuntimeError):
    """A configured research resource boundary was crossed."""


class Resource_budget:
    """Own resource measurements for one trainer process and its children."""

    def __init__(self, *, min_available_memory_mb: float = 0.0,
                 max_process_memory_mb: float = 0.0,
                 max_seconds: float = 0.0) -> None:
        self.limits = {
            "min_available_memory_mb": min_available_memory_mb,
            "max_process_memory_mb": max_process_memory_mb,
            "max_seconds": max_seconds,
        }
        for name, value in self.limits.items():
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.started = time.perf_counter()
        self.peak_process_memory_mb = 0.0
        self.minimum_available_memory_mb = None
        self.last_measurement: Dict[str, Any] = {}

    def check(self) -> Dict[str, Any]:
        elapsed = time.perf_counter() - self.started
        measurement: Dict[str, Any] = {"elapsed_seconds": elapsed}
        if (self.limits["min_available_memory_mb"] or
                self.limits["max_process_memory_mb"]):
            import psutil

            process = psutil.Process()
            memory_bytes = process.memory_info().rss
            for child in process.children(recursive=True):
                try:
                    memory_bytes += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            process_mb = memory_bytes / 1024 ** 2
            available_mb = psutil.virtual_memory().available / 1024 ** 2
            self.peak_process_memory_mb = max(self.peak_process_memory_mb, process_mb)
            self.minimum_available_memory_mb = (
                available_mb if self.minimum_available_memory_mb is None else
                min(self.minimum_available_memory_mb, available_mb))
            measurement.update(process_memory_mb=process_mb,
                               available_memory_mb=available_mb)
        self.last_measurement = measurement
        reason = None
        if (self.limits["max_seconds"] and elapsed >= self.limits["max_seconds"]):
            reason = "elapsed_time_limit"
        if (self.limits["max_process_memory_mb"] and
                measurement["process_memory_mb"] > self.limits["max_process_memory_mb"]):
            reason = "process_tree_memory_limit"
        if (self.limits["min_available_memory_mb"] and
                measurement["available_memory_mb"] < self.limits["min_available_memory_mb"]):
            reason = "system_memory_reserve"
        if reason:
            raise Resource_limit_exceeded(f"{reason}: {measurement}")
        return measurement

    def summary(self) -> Dict[str, Any]:
        return {"limits": dict(self.limits),
                "peak_process_memory_mb": self.peak_process_memory_mb,
                "minimum_available_memory_mb": self.minimum_available_memory_mb,
                "last_measurement": dict(self.last_measurement)}
