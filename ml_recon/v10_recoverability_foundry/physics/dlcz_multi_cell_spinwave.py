"""Pure-NumPy multi-cell spin-wave channel for unbound DLCZ diagnostics.

The channel describes a collective cold-ensemble spin wave, not four detector
paths: an AOD analyzer projects the cell amplitudes into one selected optical
mode, which only later reaches the splitter/SPAD renderer.  Its parameters are
deliberately hypothesis-only until an experiment-physics receipt binds them.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


MULTI_CELL_SPINWAVE_STATUS = "HYPOTHESIS_UNBOUND"


def _normalized_complex_mode(values: Sequence[complex], *, label: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.complex128).reshape(-1)
    if vector.size < 1 or not np.isfinite(vector).all():
        raise ValueError(f"{label} must be a finite non-empty complex vector")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError(f"{label} must have nonzero norm")
    return np.ascontiguousarray(vector / norm, dtype=np.complex128)


def calibrated_write_mode(
    mode_probability: Sequence[float],
    *,
    amplitude_error: Sequence[float],
    phase_error: Sequence[float],
    crosstalk: Sequence[float],
    common_phase_rad: float = 0.0,
) -> np.ndarray:
    """Build an unbound collective write mode without discarding phase.

    ``amplitude_error`` is explicitly interpreted as a *log-intensity*
    sensitivity, so the pre-crosstalk cell intensity is
    ``mode_probability * exp(amplitude_error)``.  ``crosstalk[i]`` then
    replaces that fraction of cell ``i``'s original intensity by the uniform
    mean of the other original cells.  The map is evaluated from one immutable
    input vector; it is not a sequential calibration update and is therefore
    independent of cell enumeration order.  All quantities remain
    ``HYPOTHESIS_UNBOUND`` until an apparatus receipt defines their convention.
    """

    probability = np.asarray(mode_probability, dtype=np.float64).reshape(-1)
    amplitude = np.asarray(amplitude_error, dtype=np.float64).reshape(-1)
    phase = np.asarray(phase_error, dtype=np.float64).reshape(-1)
    mixing = np.asarray(crosstalk, dtype=np.float64).reshape(-1)
    if (
        probability.size < 1
        or probability.shape != amplitude.shape
        or probability.shape != phase.shape
        or probability.shape != mixing.shape
        or not np.isfinite(probability).all()
        or not np.isfinite(amplitude).all()
        or not np.isfinite(phase).all()
        or not np.isfinite(mixing).all()
        or np.any(probability < 0.0)
        or not np.isclose(float(probability.sum()), 1.0)
        or np.any((mixing < 0.0) | (mixing >= 1.0))
        or not math.isfinite(float(common_phase_rad))
    ):
        raise ValueError("write-mode calibration inputs are invalid")
    pre_crosstalk_intensity = probability * np.exp(np.clip(amplitude, -20.0, 20.0))
    if probability.size == 1:
        calibrated_probability = pre_crosstalk_intensity.copy()
    else:
        other_mean = (
            float(pre_crosstalk_intensity.sum()) - pre_crosstalk_intensity
        ) / (probability.size - 1)
        calibrated_probability = (
            1.0 - mixing
        ) * pre_crosstalk_intensity + mixing * other_mean
    if float(calibrated_probability.sum()) <= 0.0:
        raise ValueError("write-mode calibration has zero mass")
    calibrated_probability /= calibrated_probability.sum()
    return _normalized_complex_mode(
        np.sqrt(calibrated_probability)
        * np.exp(1j * (phase + float(common_phase_rad))),
        label="calibrated write mode",
    )


def aod_analyzer_mode(
    amplitudes: Sequence[float], phases: Sequence[float]
) -> np.ndarray:
    """Return ``normalize(a_o exp(-i phi_o))`` for one registered operation."""

    amplitude = np.asarray(amplitudes, dtype=np.float64).reshape(-1)
    phase = np.asarray(phases, dtype=np.float64).reshape(-1)
    if (
        amplitude.size < 1
        or amplitude.shape != phase.shape
        or not np.isfinite(amplitude).all()
        or not np.isfinite(phase).all()
        or np.any(amplitude < 0.0)
    ):
        raise ValueError("AOD analyzer amplitudes/phases are invalid")
    return _normalized_complex_mode(
        amplitude * np.exp(-1j * phase), label="AOD analyzer mode"
    )


def unit_correlation_kernel(n_cell: int) -> np.ndarray:
    """Return the maximally coherent PSD correlation kernel for a sensitivity law."""

    if type(n_cell) is not int or n_cell < 1:
        raise ValueError("n_cell must be a positive integer")
    return np.ones((n_cell, n_cell), dtype=np.complex128)


def _validated_correlation_kernel(
    kernel: Sequence[Sequence[complex]], *, n_cell: int
) -> np.ndarray:
    matrix = np.asarray(kernel, dtype=np.complex128)
    if matrix.shape != (n_cell, n_cell) or not np.isfinite(matrix).all():
        raise ValueError("storage coherence kernel must be finite and cell-aligned")
    if not np.allclose(matrix, matrix.conj().T, atol=1.0e-12, rtol=0.0):
        raise ValueError("storage coherence kernel must be Hermitian")
    if not np.allclose(np.diag(matrix), 1.0, atol=1.0e-12, rtol=0.0):
        raise ValueError("storage coherence kernel must have unit diagonal")
    if float(np.linalg.eigvalsh(matrix).min(initial=0.0)) < -1.0e-12:
        raise ValueError("storage coherence kernel must be positive semidefinite")
    return np.ascontiguousarray(matrix, dtype=np.complex128)


@dataclass(frozen=True)
class SpinwaveProjection:
    """Selected-mode result before any splitter or detector efficiency."""

    density_matrix: np.ndarray
    storage_kernel: np.ndarray
    selected_mode_probability: float
    selected_pre_splitter_probability: float
    status: str = MULTI_CELL_SPINWAVE_STATUS


@dataclass(frozen=True)
class DlczMultiCellSpinwaveChannel:
    """A PSD one-particle channel plus the operation AOD read projection.

    ``survival_by_cell`` controls diagonal population survival.  The separately
    bounded ``off_diagonal_visibility`` only attenuates inter-cell coherence;
    neither quantity is silently substituted for the other.  ``optical_path``
    is applied once, after the selected AOD port and before the fixed splitter.
    """

    write_mode: Sequence[complex]
    analyzer_mode: Sequence[complex]
    survival_by_cell: Sequence[float]
    storage_coherence_kernel: Sequence[Sequence[complex]]
    off_diagonal_visibility: float = 1.0
    read_conversion_efficiency: float = 1.0
    optical_path_efficiency: float = 1.0
    parameter_status: str = MULTI_CELL_SPINWAVE_STATUS

    def validate(self) -> None:
        write = _normalized_complex_mode(self.write_mode, label="write mode")
        analyzer = _normalized_complex_mode(self.analyzer_mode, label="analyzer mode")
        if write.shape != analyzer.shape:
            raise ValueError("write and analyzer modes must have equal cell count")
        survival = np.asarray(self.survival_by_cell, dtype=np.float64).reshape(-1)
        if (
            survival.shape != write.shape
            or not np.isfinite(survival).all()
            or np.any((survival < 0.0) | (survival > 1.0))
        ):
            raise ValueError("per-cell survival must be finite probabilities")
        _validated_correlation_kernel(self.storage_coherence_kernel, n_cell=write.size)
        for name in (
            "off_diagonal_visibility",
            "read_conversion_efficiency",
            "optical_path_efficiency",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.parameter_status != MULTI_CELL_SPINWAVE_STATUS:
            raise ValueError(
                "multi-cell spin-wave channel must remain HYPOTHESIS_UNBOUND"
            )

    @property
    def n_cell(self) -> int:
        self.validate()
        return len(self.write_mode)

    def normalized_write_mode(self) -> np.ndarray:
        self.validate()
        return _normalized_complex_mode(self.write_mode, label="write mode")

    def normalized_analyzer_mode(self) -> np.ndarray:
        self.validate()
        return _normalized_complex_mode(self.analyzer_mode, label="analyzer mode")

    def effective_storage_kernel(self) -> np.ndarray:
        """Return the PSD survival/dephasing Gram kernel for the cell basis."""

        self.validate()
        survival = np.asarray(self.survival_by_cell, dtype=np.float64)
        correlation = _validated_correlation_kernel(
            self.storage_coherence_kernel, n_cell=survival.size
        ).copy()
        if survival.size > 1:
            off_diagonal = ~np.eye(survival.size, dtype=bool)
            correlation[off_diagonal] *= float(self.off_diagonal_visibility)
        kernel = np.sqrt(np.outer(survival, survival)) * correlation
        # This is a congruence and Schur product of PSD kernels.  Keep the
        # numerical assertion local rather than clipping an invalid kernel.
        if float(np.linalg.eigvalsh(kernel).min(initial=0.0)) < -1.0e-12:
            raise RuntimeError("constructed storage kernel is not PSD")
        return np.ascontiguousarray(kernel, dtype=np.complex128)

    def project_single_excitation(self) -> SpinwaveProjection:
        """Propagate the normalized write mode and project the AOD read mode."""

        write = self.normalized_write_mode()
        analyzer = self.normalized_analyzer_mode()
        kernel = self.effective_storage_kernel()
        density = kernel * np.outer(write, write.conj())
        selected = float(np.real(np.vdot(analyzer, density @ analyzer)))
        # A PSD density and normalized analyzer make only roundoff correction
        # necessary; an appreciable violation is an invalid channel.
        if selected < -1.0e-12 or selected > 1.0 + 1.0e-12:
            raise RuntimeError("selected AOD probability lies outside [0, 1]")
        selected = min(1.0, max(0.0, selected))
        return SpinwaveProjection(
            density_matrix=np.ascontiguousarray(density, dtype=np.complex128),
            storage_kernel=kernel,
            selected_mode_probability=selected,
            selected_pre_splitter_probability=(
                float(self.read_conversion_efficiency)
                * float(self.optical_path_efficiency)
                * selected
            ),
        )

    def n2_coherent_amplitudes(
        self,
    ) -> tuple[dict[int, complex], dict[tuple[int, int], complex]]:
        """Return named local and cross amplitudes of ``(S_write†)^2/sqrt(2)``."""

        write = self.normalized_write_mode()
        local = {index: complex(value * value) for index, value in enumerate(write)}
        cross = {
            (left, right): complex(math.sqrt(2.0) * write[left] * write[right])
            for left in range(write.size)
            for right in range(left + 1, write.size)
        }
        normalization = sum(abs(value) ** 2 for value in local.values()) + sum(
            abs(value) ** 2 for value in cross.values()
        )
        if not math.isclose(normalization, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise RuntimeError("n=2 coherent amplitudes are not normalized")
        return local, cross


__all__ = [
    "MULTI_CELL_SPINWAVE_STATUS",
    "DlczMultiCellSpinwaveChannel",
    "SpinwaveProjection",
    "aod_analyzer_mode",
    "calibrated_write_mode",
    "unit_correlation_kernel",
]
