"""Baseline-subtracting counter with overflow carry handling.

The data UI reads cumulative counters from the FPGA serial stream. Each
counter is a small fixed-width unsigned integer that wraps when it overflows:

  - signal_bins: 9 bits per cell (256 cells)
  - idler_bins:   9 bits per cell
  - coin_bins:    6 bits per cell
  - signal_hist / idler_hist (cellhist): 8 bits per bin (10 bins)
  - read_times / write_seq: 40 bits each
  - (and more, see ui_4qubit.py / ui_3qubit.py)

For a 4-bit bin (modulus=16) the wrap can produce very small decreases (e.g.
``0011 -> 0000`` is a drop of 3), so the legacy ``overflow_sensitivity=10``
heuristic misses them. ``BaselineCounter`` implements the user-requested
semantics:

  1. At Start: capture ``baseline = corrected_value`` (the value AFTER any
     wraps already accumulated).
  2. On each update: detect wrap (raw < last_raw by > sensitivity), bump the
     carry, then return ``(raw + carry*modulus) - baseline``.
  3. The UI starts at 0 at Start time and only ever grows as time passes.

Example (4-bit modulus=16, baseline at start=1, sensitivity=1):
  reading 0001 (start) -> display 0
  reading 0011 -> 3 > 1, no wrap -> display 2
  reading 0000 -> 0 < 3, drop=3>1 -> carry=1, display (16+0)-1 = 15
  reading 0001 -> 1 > 0, no wrap -> display 16
  ...
  reading 1111 -> display 30
  reading 0000 -> 0 < 15, wrap -> carry=2, display (32+0)-1 = 31
"""

from __future__ import annotations

from typing import Optional


class BaselineCounter:
    """A single baseline-subtracting counter that survives wrap-around.

    Args:
        modulus: Value at which the underlying counter wraps (e.g., 16 for a
            4-bit bin, 256 for 8-bit, 512 for 9-bit).
        sensitivity: How large a backward step must be to count as a wrap
            rather than noise. Use 1 for low-bit counters (4-/6-/8-/9-bit) so
            even small drops like 0011->0000 (drop=3) are detected. Default 10
            matches the legacy read_loop behaviour for the 40-bit read_times /
            write_seq.
        name: Optional human label for debug logging.
    """

    __slots__ = ("modulus", "sensitivity", "name", "_carry", "_last_raw", "_baseline")

    def __init__(self, modulus: int, sensitivity: int = 10, name: Optional[str] = None):
        if modulus <= 0:
            raise ValueError(f"modulus must be > 0 (got {modulus})")
        if sensitivity < 0:
            raise ValueError(f"sensitivity must be >= 0 (got {sensitivity})")
        self.modulus = modulus
        self.sensitivity = sensitivity
        self.name = name or f"counter(mod={modulus})"
        self._carry: int = 0
        self._last_raw: Optional[int] = None
        self._baseline: Optional[int] = None  # corrected value at start

    def reset(self) -> None:
        """Clear all state. Call before a new measurement window."""
        self._carry = 0
        self._last_raw = None
        self._baseline = None

    def set_baseline_from_current(self) -> int:
        """Snapshot the current corrected value as the baseline.

        Must be called *after* at least one ``update()`` (so that ``_last_raw``
        is known and the carry reflects any prior wraps). Returns the captured
        baseline so the caller can log it.
        """
        if self._last_raw is None:
            raise RuntimeError(
                f"BaselineCounter({self.name}): cannot set baseline before any update()"
            )
        corrected = self._last_raw + self._carry * self.modulus
        self._baseline = corrected
        return corrected

    @property
    def baseline(self) -> Optional[int]:
        return self._baseline

    @property
    def corrected(self) -> Optional[int]:
        """Latest corrected (carry-aware) value, or None if no update yet."""
        if self._last_raw is None:
            return None
        return self._last_raw + self._carry * self.modulus

    @property
    def carry(self) -> int:
        return self._carry

    def update(self, raw: int) -> int:
        """Feed a new raw reading and return the baseline-subtracted display.

        The returned value is guaranteed to never decrease between successive
        calls (provided the underlying counter is monotonically non-decreasing
        modulo ``modulus``), which is the property the UI relies on.
        """
        raw = int(raw)
        if self._last_raw is None:
            # First reading after start (or after reset): seed.
            self._last_raw = raw
        elif raw < self._last_raw:
            drop = self._last_raw - raw
            if drop > self.sensitivity:
                # Real wrap (modulus boundary crossed).
                self._carry += 1
                self._last_raw = raw
            # else: drop within sensitivity -> treat as serial noise,
            # do not advance _last_raw, do not bump carry. The display
            # therefore stays the same as the previous frame.
        else:
            # raw >= _last_raw: normal forward growth, advance the cursor.
            self._last_raw = raw
        corrected = self._last_raw + self._carry * self.modulus
        if self._baseline is None:
            # No Start baseline captured yet -> show absolute corrected value.
            return corrected
        return corrected - self._baseline

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return (
            f"BaselineCounter(name={self.name}, modulus={self.modulus}, "
            f"sensitivity={self.sensitivity}, carry={self._carry}, "
            f"last_raw={self._last_raw}, baseline={self._baseline})"
        )