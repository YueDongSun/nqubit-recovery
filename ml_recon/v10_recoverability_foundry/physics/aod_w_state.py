"""Immutable compact single-excitation W-state representation."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _immutable_vector(
    value: object, dtype: np.dtype, *, allow_boolean: bool = False
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind == "b" and not allow_boolean:
        raise ValueError("Boolean values are not valid state data")
    array = np.ascontiguousarray(array, dtype=dtype).reshape(-1)
    return np.frombuffer(array.tobytes(), dtype=array.dtype)


@dataclass(frozen=True)
class Aod_w_state:
    coefficients: np.ndarray
    mode_mask: np.ndarray

    def __post_init__(self) -> None:
        raw_mask = np.asarray(self.mode_mask)
        if raw_mask.dtype != np.bool_:
            raise ValueError("mode_mask must have strict Boolean dtype")
        mode_mask = _immutable_vector(raw_mask, np.dtype(np.bool_), allow_boolean=True)
        coefficients = _immutable_vector(self.coefficients, np.dtype(np.complex128))
        if (
            coefficients.size < 3
            or coefficients.size != mode_mask.size
            or not np.all(np.isfinite(coefficients))
        ):
            raise ValueError("AOD W state needs finite aligned coefficients and mask")
        if not np.any(mode_mask):
            raise ValueError("at least one AOD mode must be active")
        if np.any(coefficients[~mode_mask] != 0.0):
            raise ValueError("inactive AOD modes must have exact zero coefficients")
        norm = float(np.vdot(coefficients, coefficients).real)
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("AOD W coefficients cannot be all zero")
        normalized = np.ascontiguousarray(
            coefficients / np.sqrt(norm), dtype=np.complex128
        )
        object.__setattr__(
            self,
            "coefficients",
            np.frombuffer(normalized.tobytes(), dtype=np.complex128),
        )
        object.__setattr__(self, "mode_mask", mode_mask)

    @property
    def n_mode(self) -> int:
        return int(self.coefficients.size)

    @property
    def intrinsic_dimension(self) -> int:
        return 2 * int(np.sum(self.mode_mask)) - 2

    def density_matrix(self) -> np.ndarray:
        matrix = np.ascontiguousarray(
            np.outer(self.coefficients, self.coefficients.conj()), dtype=np.complex128
        )
        return np.frombuffer(matrix.tobytes(), dtype=np.complex128).reshape(
            matrix.shape
        )
