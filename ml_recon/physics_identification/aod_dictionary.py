"""Unit-labelled single-tone and pairwise AOD candidate dictionary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from .response_forms import (
    cubic_spline_hinges,
    exponential_compression,
    hill_compression,
    rational_compression,
)


DEFAULT_SPLINE_KNOTS_MHZ = (-5.0, 0.0, 5.0)
DEFAULT_SPACING_SCALES_MHZ = (1.0, 2.5, 5.0, 10.0)


@dataclass(frozen=True)
class DictionaryDesignV1:
    matrix: np.ndarray
    feature_names: Tuple[str, ...]
    units: Tuple[str, ...]
    groups: Tuple[str, ...]


class TonePairKernel:
    """Aggregate symmetric pair interactions without inventing missing tones."""

    def __init__(self, spacing_scales_mhz: Sequence[float] = DEFAULT_SPACING_SCALES_MHZ) -> None:
        self.spacing_scales_mhz = tuple(float(value) for value in spacing_scales_mhz)

    def transform(
        self,
        frequency_mhz: Any,
        amplitude: Any,
        phase_rad: Any,
        valid_tone_mask: Any,
    ) -> Dict[str, np.ndarray]:
        frequency = np.asarray(frequency_mhz, dtype=float)
        amplitude = np.asarray(amplitude, dtype=float)
        phase = np.asarray(phase_rad, dtype=float)
        valid = np.asarray(valid_tone_mask, dtype=bool)
        if not (frequency.shape == amplitude.shape == phase.shape == valid.shape):
            raise ValueError("tone arrays and valid_tone_mask must share a shape")
        if frequency.ndim != 2:
            raise ValueError("tone arrays must have shape [samples, tones]")
        sample_count, tone_count = frequency.shape
        values: Dict[str, np.ndarray] = {
            "pair_partner_amplitude_squared": np.zeros(sample_count),
            "pair_amplitude_product": np.zeros(sample_count),
            "pair_cos_phase_difference": np.zeros(sample_count),
            "pair_sin_phase_difference": np.zeros(sample_count),
        }
        for scale in self.spacing_scales_mhz:
            values[f"pair_spacing_rbf_{scale:g}mhz"] = np.zeros(sample_count)
        pair_count = np.zeros(sample_count)
        for first in range(tone_count):
            for second in range(first + 1, tone_count):
                active = valid[:, first] & valid[:, second]
                delta_phase = phase[:, first] - phase[:, second]
                spacing = np.abs(frequency[:, first] - frequency[:, second])
                values["pair_partner_amplitude_squared"] += active * (
                    amplitude[:, first] ** 2 + amplitude[:, second] ** 2
                )
                values["pair_amplitude_product"] += active * amplitude[:, first] * amplitude[:, second]
                values["pair_cos_phase_difference"] += active * np.cos(delta_phase)
                values["pair_sin_phase_difference"] += active * np.sin(delta_phase)
                for scale in self.spacing_scales_mhz:
                    values[f"pair_spacing_rbf_{scale:g}mhz"] += active * np.exp(
                        -0.5 * np.square(spacing / max(scale, 1e-12))
                    )
                pair_count += active
        denominator = np.maximum(pair_count, 1.0)
        return {name: value / denominator for name, value in values.items()}


class AODPhysicsDictionary:
    """Build a trainable design matrix with explicit coefficient units."""

    def __init__(
        self,
        *,
        spline_knots_mhz: Sequence[float] = DEFAULT_SPLINE_KNOTS_MHZ,
        spacing_scales_mhz: Sequence[float] = DEFAULT_SPACING_SCALES_MHZ,
    ) -> None:
        self.spline_knots_mhz = tuple(float(value) for value in spline_knots_mhz)
        self.pair_kernel = TonePairKernel(spacing_scales_mhz)

    def design_matrix(self, inputs: Mapping[str, Any]) -> DictionaryDesignV1:
        frequency = np.asarray(inputs["absolute_frequency_mhz"], dtype=float)
        amplitude = np.asarray(inputs["amplitude"], dtype=float)
        phase = np.asarray(inputs.get("phase_rad", np.zeros_like(amplitude)), dtype=float)
        valid = np.asarray(inputs.get("valid_tone_mask", np.ones_like(amplitude)), dtype=bool)
        if frequency.ndim == 1:
            frequency = frequency[:, None]
            amplitude = amplitude[:, None]
            phase = phase[:, None]
            valid = valid[:, None]
        if not (frequency.shape == amplitude.shape == phase.shape == valid.shape):
            raise ValueError("AOD tone inputs must share shape [samples, tones]")
        valid_float = valid.astype(float)
        tone_count = np.maximum(valid_float.sum(axis=1), 1.0)
        mean_frequency = (frequency * valid_float).sum(axis=1) / tone_count
        total_power = np.square(amplitude * valid_float).sum(axis=1)
        mean_amp = (amplitude * valid_float).sum(axis=1) / tone_count
        features: list[np.ndarray] = [
            np.ones(frequency.shape[0]),
            mean_frequency,
            np.square(mean_frequency),
        ]
        names = ["intercept", "absolute_frequency_linear", "absolute_frequency_quadratic"]
        units = ["log_rate", "log_rate/MHz", "log_rate/MHz^2"]
        groups = ["intercept", "frequency", "frequency"]
        hinges = cubic_spline_hinges(mean_frequency, self.spline_knots_mhz)
        for index, knot in enumerate(self.spline_knots_mhz):
            features.append(hinges[:, index])
            names.append(f"absolute_frequency_spline_{knot:g}mhz")
            units.append("log_rate/MHz^3")
            groups.append("frequency_spline")
        compression = {
            "power_rational": rational_compression(total_power),
            "power_exponential": exponential_compression(total_power),
            "power_hill": hill_compression(total_power),
        }
        for name, value in compression.items():
            features.append(value)
            names.append(name)
            units.append("log_rate/power_unit")
            groups.append("power_compression")
        pair_values = self.pair_kernel.transform(frequency, amplitude, phase, valid)
        for name, value in pair_values.items():
            features.append(value)
            names.append(name)
            units.append("log_rate/pair_unit")
            groups.append("tone_pair")
        phase_mean_sin = (np.sin(phase) * valid_float).sum(axis=1) / tone_count
        phase_mean_cos = (np.cos(phase) * valid_float).sum(axis=1) / tone_count
        features.extend((mean_amp * phase_mean_sin, mean_amp * phase_mean_cos))
        names.extend(("amplitude_phase_sin_coupling", "amplitude_phase_cos_coupling"))
        units.extend(("log_rate/amplitude", "log_rate/amplitude"))
        groups.extend(("amplitude_phase", "amplitude_phase"))
        optional_terms = (
            ("write_read_path", "write_read_path_imbalance", "log_rate/path_unit", "path"),
            ("cycle_index", "cycle_read_residual", "log_rate/cycle", "cycle"),
            ("spatial_coordinate", "spatial_grid_residual", "log_rate/grid_unit", "spatial"),
            ("time_hours", "time_drift", "log_rate/hour", "temporal"),
            ("temperature_c", "temperature_drift", "log_rate/degC", "temporal"),
            ("detector_nuisance", "detector_nuisance", "log_rate/nuisance_unit", "detector_source"),
            ("source_nuisance", "source_nuisance", "log_rate/nuisance_unit", "detector_source"),
        )
        for input_name, feature_name, unit, group in optional_terms:
            if input_name in inputs:
                value = np.asarray(inputs[input_name], dtype=float).reshape(frequency.shape[0], -1).mean(axis=1)
                features.append(value)
                names.append(feature_name)
                units.append(unit)
                groups.append(group)
        return DictionaryDesignV1(
            matrix=np.stack(features, axis=-1),
            feature_names=tuple(names),
            units=tuple(units),
            groups=tuple(groups),
        )


__all__ = [
    "AODPhysicsDictionary",
    "DEFAULT_SPACING_SCALES_MHZ",
    "DEFAULT_SPLINE_KNOTS_MHZ",
    "DictionaryDesignV1",
    "TonePairKernel",
]
