"""V9 rho8 operation estimators (charter section 9.1).

Two and only two official operation estimators:

- ``cs_minus_ir``: signed ``C_o/S_o - I_o/R_o`` (zero when S_o<=0 or R_o<=0), with the
  signed value preserved as a diagnostic and the nonnegative value used only at the
  reconstruction boundary.
- ``cs``: ``C_o/S_o`` (zero when S_o<=0), retaining the zero-signal mask.

No coin/idler or shifted-to-zero estimator may substitute for these two official
estimators in the canonical rho8 set (legacy diagnostics stay outside canonical views).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

SYSTEM_OPERATION_COUNTS = {"3qubit": 75, "4qubit": 256}
ESTIMATOR_KEYS = ("cs_minus_ir", "cs")


@dataclass(frozen=True)
class V9OperationEstimatorResultV1:
    """Output of one official operation estimator.

    ``x`` is the nonnegative value used for density-matrix reconstruction.
    ``signed_x`` is the signed diagnostic (identical to ``x`` for ``cs``).
    """

    estimator_key: str
    system_type: str
    x: np.ndarray
    signed_x: np.ndarray
    zero_signal_mask: np.ndarray
    zero_read_mask: np.ndarray  # all False for ``cs``
    negative_fraction: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimator_key": self.estimator_key,
            "system_type": self.system_type,
            "negative_fraction": float(self.negative_fraction),
            "zero_signal_count": int(np.sum(self.zero_signal_mask)),
            "zero_read_count": int(np.sum(self.zero_read_mask)),
        }


def _validate_counts(system_type: str, name: str, values: np.ndarray) -> np.ndarray:
    expected = SYSTEM_OPERATION_COUNTS[system_type]
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.size != expected:
        raise ValueError(f"{name} must contain {expected} operations, got {result.size}")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def build_operation_estimator(
    system_type: str,
    estimator_key: str,
    *,
    signal_counts: np.ndarray,
    idler_counts: np.ndarray,
    coin_counts: np.ndarray,
    read_count: np.ndarray | float,
) -> V9OperationEstimatorResultV1:
    """Apply one official estimator to per-operation count arrays."""
    if estimator_key not in ESTIMATOR_KEYS:
        raise ValueError(f"estimator_key must be one of {ESTIMATOR_KEYS}, got {estimator_key}")
    signal = _validate_counts(system_type, "signal_counts", signal_counts)
    idler = _validate_counts(system_type, "idler_counts", idler_counts)
    coin = _validate_counts(system_type, "coin_counts", coin_counts)
    read = np.asarray(read_count, dtype=np.float64).reshape(-1)
    if read.size == 1:
        read = np.full(signal.size, float(read[0]), dtype=np.float64)
    if read.size != signal.size or not np.all(np.isfinite(read)) or np.any(read <= 0.0):
        raise ValueError("read_count must be positive scalar or one value per operation")

    zero_signal_mask = signal <= 0.0
    zero_read_mask = read <= 0.0

    coin_over_signal = np.divide(
        coin,
        signal,
        out=np.zeros_like(coin),
        where=signal > 0.0,
    )

    if estimator_key == "cs_minus_ir":
        idler_over_read = np.divide(
            idler,
            read,
            out=np.zeros_like(idler),
            where=read > 0.0,
        )
        signed_x = np.where(zero_signal_mask | zero_read_mask, 0.0, coin_over_signal - idler_over_read)
        negative_fraction = float(np.mean(signed_x < 0.0)) if signal.size else 0.0
    else:  # cs
        signed_x = np.where(zero_signal_mask, 0.0, coin_over_signal)
        negative_fraction = 0.0

    x = np.maximum(signed_x, 0.0)
    return V9OperationEstimatorResultV1(
        estimator_key=estimator_key,
        system_type=system_type,
        x=x,
        signed_x=signed_x,
        zero_signal_mask=zero_signal_mask,
        zero_read_mask=zero_read_mask if estimator_key == "cs_minus_ir" else np.zeros_like(zero_read_mask),
        negative_fraction=negative_fraction,
    )


def build_all_estimators(
    system_type: str,
    *,
    signal_counts: np.ndarray,
    idler_counts: np.ndarray,
    coin_counts: np.ndarray,
    read_count: np.ndarray | float,
) -> Dict[str, V9OperationEstimatorResultV1]:
    """Build both official estimators."""
    return {
        key: build_operation_estimator(
            system_type, key,
            signal_counts=signal_counts, idler_counts=idler_counts,
            coin_counts=coin_counts, read_count=read_count,
        )
        for key in ESTIMATOR_KEYS
    }
