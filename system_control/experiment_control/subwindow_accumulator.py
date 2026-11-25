"""Authoritative fixed-duration subwindows for V6 feedback episodes.

The accumulator is a software-side view over the existing FPGA acquisition
stream.  It does not change the FPGA protocol.  A normal feedback measurement
is authoritative only after six 150 second records complete the 900 second
window.  Early completion is reserved for an explicitly recorded serious
safety failure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


DEFAULT_MEASUREMENT_WINDOW_S = 900.0
DEFAULT_SUBWINDOW_S = 150.0
SUBWINDOW_DURATION_TOLERANCE_S = 1e-6


@dataclass(frozen=True)
class SubwindowRecordV1:
    subwindow_index: int
    started_at: float
    ended_at: float
    duration_s: float
    count_arrays: Mapping[str, Any]
    read_count: int
    write_count: int
    multi_counts: Mapping[str, int] = field(default_factory=dict)
    temperature_c: Optional[float] = None
    quality_flags: Tuple[str, ...] = ()
    serious_safety_failure: bool = False

    def __post_init__(self) -> None:
        if self.subwindow_index < 1:
            raise ValueError("subwindow_index must be positive")
        if self.ended_at < self.started_at:
            raise ValueError("subwindow ended_at must be >= started_at")
        if self.duration_s <= 0.0:
            raise ValueError("subwindow duration_s must be positive")
        if abs((self.ended_at - self.started_at) - self.duration_s) > SUBWINDOW_DURATION_TOLERANCE_S:
            raise ValueError("subwindow timestamps do not match duration_s")
        if self.read_count < 0 or self.write_count < 0:
            raise ValueError("read_count and write_count must be non-negative")
        for name, value in self.multi_counts.items():
            if int(value) < 0:
                raise ValueError(f"multi count must be non-negative: {name}")
        for name, value in self.count_arrays.items():
            array = np.asarray(value)
            if array.size == 0 or not np.all(np.isfinite(array)):
                raise ValueError(f"count array must be finite and non-empty: {name}")
            if np.any(array < 0):
                raise ValueError(f"count array must be non-negative: {name}")

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["count_arrays"] = {
            str(name): np.asarray(value).tolist()
            for name, value in self.count_arrays.items()
        }
        payload["multi_counts"] = {
            str(name): int(value) for name, value in self.multi_counts.items()
        }
        payload["quality_flags"] = list(self.quality_flags)
        return payload

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "SubwindowRecordV1":
        values = dict(payload)
        values["quality_flags"] = tuple(str(item) for item in values.get("quality_flags", ()))
        return cls(**values)


class SubwindowAccumulator:
    """Collect sequential subwindows and enforce the authoritative time contract."""

    def __init__(
        self,
        measurement_window_s: float = DEFAULT_MEASUREMENT_WINDOW_S,
        subwindow_s: float = DEFAULT_SUBWINDOW_S,
    ) -> None:
        self.measurement_window_s = float(measurement_window_s)
        self.subwindow_s = float(subwindow_s)
        if self.measurement_window_s <= 0.0 or self.subwindow_s <= 0.0:
            raise ValueError("measurement and subwindow durations must be positive")
        ratio = self.measurement_window_s / self.subwindow_s
        if abs(ratio - round(ratio)) > SUBWINDOW_DURATION_TOLERANCE_S:
            raise ValueError("measurement_window_s must be divisible by subwindow_s")
        self.expected_subwindows = int(round(ratio))
        self._records: list[SubwindowRecordV1] = []

    @property
    def records(self) -> Tuple[SubwindowRecordV1, ...]:
        return tuple(self._records)

    @property
    def serious_safety_failure(self) -> bool:
        return any(record.serious_safety_failure for record in self._records)

    @property
    def authoritative_complete(self) -> bool:
        return (
            not self.serious_safety_failure
            and len(self._records) == self.expected_subwindows
            and abs(self.total_duration_s - self.measurement_window_s)
            <= SUBWINDOW_DURATION_TOLERANCE_S
        )

    @property
    def total_duration_s(self) -> float:
        return float(sum(record.duration_s for record in self._records))

    def add(self, record: SubwindowRecordV1) -> None:
        if self.serious_safety_failure:
            raise RuntimeError("cannot append after a serious safety abort")
        expected_index = len(self._records) + 1
        if record.subwindow_index != expected_index:
            raise ValueError(f"expected subwindow_index={expected_index}")
        if len(self._records) >= self.expected_subwindows:
            raise ValueError("measurement already contains all expected subwindows")
        if (
            not record.serious_safety_failure
            and abs(record.duration_s - self.subwindow_s) > SUBWINDOW_DURATION_TOLERANCE_S
        ):
            raise ValueError("ordinary subwindow must complete the fixed duration")
        if self._records and abs(
            record.started_at - self._records[-1].ended_at
        ) > SUBWINDOW_DURATION_TOLERANCE_S:
            raise ValueError("subwindow timestamps must be contiguous")
        self._records.append(record)

    def require_authoritative_window(self) -> None:
        if self.serious_safety_failure:
            raise RuntimeError("measurement ended on a serious safety failure")
        if not self.authoritative_complete:
            raise RuntimeError(
                f"authoritative measurement requires {self.expected_subwindows} subwindows "
                f"and {self.measurement_window_s:.0f} seconds"
            )

    def metadata(self) -> Dict[str, Any]:
        return {
            "schema": "SubwindowMeasurementV1",
            "measurement_window_s": self.measurement_window_s,
            "subwindow_s": self.subwindow_s,
            "expected_subwindows": self.expected_subwindows,
            "completed_subwindows": len(self._records),
            "total_duration_s": self.total_duration_s,
            "authoritative_complete": self.authoritative_complete,
            "serious_safety_failure": self.serious_safety_failure,
            "records": [record.to_json_dict() for record in self._records],
        }


__all__ = [
    "DEFAULT_MEASUREMENT_WINDOW_S",
    "DEFAULT_SUBWINDOW_S",
    "SubwindowAccumulator",
    "SubwindowRecordV1",
]
