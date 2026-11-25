"""A bounded AOD point-response digital twin.

This module is numerically evaluable with NumPy and intentionally keeps the
physical assumptions small: unknown calibration quantities are neutral for the
forward calculation but are surfaced in masks and uncertainty diagnostics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np

from ml_recon.physics_hybrid_v5.command_features import compose_aod_channel_phase

from system_control.experiment_control.operation_points import OperationPointSpec


def wrap_phase(values: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(values) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class AODResponseParameters:
    """Low-order, bounded channel response parameters.

    Coefficients may be fit from dedicated calibration data.  Defaults are an
    identity response, not invented calibration values.
    """
    channel_amplitude_gain: Dict[str, float] = field(default_factory=dict)
    channel_phase_offset_rad: Dict[str, float] = field(default_factory=dict)
    frequency_amplitude_slope_per_mhz: Dict[str, float] = field(default_factory=dict)
    frequency_phase_slope_rad_per_mhz: Dict[str, float] = field(default_factory=dict)
    compression_max: Dict[str, float] = field(default_factory=dict)
    crosstalk_max: Dict[str, float] = field(default_factory=dict)
    reference_frequency_mhz: float = 100.0


@dataclass
class PointResponse:
    actual_amplitudes: np.ndarray
    actual_phases_rad: np.ndarray
    normalized_measurement_state: np.ndarray
    measurement_projector: np.ndarray
    expected_point_efficiency: float | None
    expected_background_rate: float | None
    expected_leakage_rate: float | None
    uncertainty_diagnostics: Dict[str, Any]
    missing_value_masks: Dict[str, Any]

    def to_json_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key in ("actual_amplitudes", "actual_phases_rad", "normalized_measurement_state", "measurement_projector"):
            value = getattr(self, key)
            if np.iscomplexobj(value):
                result[key] = {"real": np.real(value).tolist(), "imag": np.imag(value).tolist()}
            else:
                result[key] = np.asarray(value).tolist()
        return result


class AODPointResponseModel:
    """Forward model for one operation point.

    Read-channel diffraction-order signs are carried by the registry.  No sign
    is globally imposed on write channels.
    """

    def __init__(self, parameters: Optional[AODResponseParameters] = None):
        self.parameters = parameters or AODResponseParameters()

    def evaluate(self, point: OperationPointSpec, calibration: Any = None) -> PointResponse:
        n = len(point.cell_indices)
        command_amp, command_phase, channels = self._projection_commands(point)
        amplitudes = np.zeros(n, dtype=float)
        phases = np.zeros(n, dtype=float)
        cell_missing = []
        group_missing = []
        for index, (coord, freq, base_amp, base_phase) in enumerate(zip(
            point.cell_coordinates, point.frequencies_mhz, command_amp, command_phase,
        )):
            row, col = coord
            cell = _calibration_item(calibration, "cell", row, col)
            group = _calibration_item(calibration, "group", point.row, point.group, point.cycle_index)
            cell_eff, missing_eff = _value(cell, "retrieval_efficiency", 1.0)
            # A readout efficiency is multiplicative only when explicitly measured.
            readout, missing_readout = _value(cell, "readout_efficiency", 1.0)
            path_eff, missing_path = _value(group, "optical_path_efficiency", 1.0)
            cell_phase, missing_cell_phase = _value(cell, "phase_offset_rad", 0.0)
            group_phase, missing_group_phase = _value(group, "shared_phase_offset_rad", 0.0)
            group_xtalk, missing_xtalk = _value(group, "crosstalk_strength", 0.0)
            total_power = _total_power(point, index)
            channel_gain = np.mean([self.parameters.channel_amplitude_gain.get(channel, 1.0) for channel in channels])
            compression = self._compression(channels, total_power, group)
            frequency_gain = self._frequency_gain(channels, freq)
            # Calibration efficiencies are measured count/probability
            # efficiencies.  A projection state is a field amplitude, so the
            # combined efficiency enters once through its square root.
            probability_efficiency = max(0.0, float(cell_eff) * float(readout) * float(path_eff))
            amplitudes[index] = max(
                0.0,
                float(base_amp)
                * np.sqrt(probability_efficiency)
                * channel_gain
                * frequency_gain
                * compression,
            )
            sign = _read_projection_sign(point, channels)
            nonlinear_phase = self._frequency_phase(channels, freq)
            cross_phase = self._crosstalk_phase(point, index, group_xtalk)
            phases[index] = float(wrap_phase(compose_aod_channel_phase(
                basis_phase_rad=base_phase,
                channel_phases_rad=(),
                diffraction_order_signs=(sign,),
                path_kind="read",
                residual_phase_rad=cell_phase + group_phase + nonlinear_phase + cross_phase,
            )))
            cell_missing.append({
                "efficiency": missing_eff or missing_readout, "phase": missing_cell_phase,
            })
            group_missing.append({"path": missing_path, "phase": missing_group_phase, "crosstalk": missing_xtalk})

        # Basis amplitudes are part of the commanded projection, not a second
        # calibration.  They preserve the current tomography convention.
        basis = np.asarray(point.basis_amplitudes, dtype=float)
        if basis.size != n:
            basis = np.ones(n, dtype=float)
        state = basis * amplitudes * np.exp(1j * phases)
        norm = float(np.linalg.norm(state))
        if norm <= 0.0:
            state = np.zeros(n, dtype=complex)
            state[0] = 1.0
            zero_norm = True
        else:
            state = state / norm
            zero_norm = False
        operation = _calibration_item(calibration, "operation", point.global_index)
        efficiency, missing_op_eff = _value(operation, "effective_efficiency", None)
        if efficiency is None:
            known_efficiencies = [
                _value(_calibration_item(calibration, "cell", r, c), "retrieval_efficiency", None)[0]
                for r, c in point.cell_coordinates
            ]
            usable = [value for value in known_efficiencies if value is not None]
            efficiency = float(np.mean(usable)) if usable else None
        background, missing_background = _value(operation, "effective_background_rate", None)
        leakage, missing_leakage = _value(operation, "effective_leakage_rate", None)
        variance_scale, missing_variance = _value(operation, "variance_scale", None)
        uncertainty = _uncertainty(point, calibration, variance_scale)
        uncertainty["zero_norm_projection"] = zero_norm
        uncertainty["efficiency_semantics"] = "probability_to_field_amplitude_sqrt"
        return PointResponse(
            actual_amplitudes=amplitudes,
            actual_phases_rad=phases,
            normalized_measurement_state=state,
            measurement_projector=np.outer(state, state.conj()),
            expected_point_efficiency=None if efficiency is None else float(efficiency),
            expected_background_rate=None if background is None else float(background),
            expected_leakage_rate=None if leakage is None else float(leakage),
            uncertainty_diagnostics=uncertainty,
            missing_value_masks={
                "cells": cell_missing, "groups": group_missing,
                "operation_efficiency": missing_op_eff, "background": missing_background,
                "leakage": missing_leakage, "variance_scale": missing_variance,
            },
        )

    def _projection_commands(self, point: OperationPointSpec) -> tuple[np.ndarray, np.ndarray, list[str]]:
        # The current read projection is encoded by R_CH1 (x) and R_CH2 (y),
        # while the basis configuration carries the intended relative phases.
        x = point.channel_commands.get("R_CH1", {})
        y = point.channel_commands.get("R_CH2", {})
        n = len(point.cell_indices)
        x_amp = _padded(x.get("amplitudes"), n, 1.0)
        x_phase = _padded(x.get("phases_rad"), n, 0.0)
        y_amp = _padded(y.get("amplitudes"), n, 1.0)
        y_phase = _padded(y.get("phases_rad"), n, 0.0)
        basis_phase = _padded(point.basis_phases_rad, n, 0.0)
        return x_amp * y_amp, basis_phase + x_phase + y_phase, ["R_CH1", "R_CH2"]

    def _compression(self, channels: list[str], total_power: float, group: Any) -> float:
        configured = np.mean([self.parameters.compression_max.get(channel, 0.0) for channel in channels])
        group_compression, missing = _value(group, "total_power_compression", 0.0)
        bounded = float(np.clip(configured + group_compression, 0.0, 0.95))
        return max(0.05, 1.0 - bounded * (total_power / (1.0 + total_power)))

    def _frequency_gain(self, channels: list[str], frequency_mhz: float) -> float:
        slope = np.mean([self.parameters.frequency_amplitude_slope_per_mhz.get(channel, 0.0) for channel in channels])
        # Exponentiation guarantees positive response while retaining a smooth fit.
        return float(np.exp(np.clip(slope * (frequency_mhz - self.parameters.reference_frequency_mhz), -3.0, 3.0)))

    def _frequency_phase(self, channels: list[str], frequency_mhz: float) -> float:
        offset = np.mean([self.parameters.channel_phase_offset_rad.get(channel, 0.0) for channel in channels])
        slope = np.mean([self.parameters.frequency_phase_slope_rad_per_mhz.get(channel, 0.0) for channel in channels])
        return float(wrap_phase(offset + slope * (frequency_mhz - self.parameters.reference_frequency_mhz)))

    def _crosstalk_phase(self, point: OperationPointSpec, index: int, group_crosstalk: float) -> float:
        configured = np.mean([self.parameters.crosstalk_max.get(channel, 0.0) for channel in ("R_CH1", "R_CH2")])
        bounded = float(np.clip(configured + group_crosstalk, -0.5, 0.5))
        other = sum(point.basis_amplitudes) - (point.basis_amplitudes[index] if index < len(point.basis_amplitudes) else 0.0)
        return bounded * float(other) / max(1, len(point.cell_indices) - 1)


def _padded(values: Any, n: int, fill: float) -> np.ndarray:
    if values is None:
        return np.full(n, fill, dtype=float)
    try:
        array = np.asarray([fill if value is None else value for value in values], dtype=float).reshape(-1)
    except TypeError:
        array = np.asarray([fill if values is None else values], dtype=float).reshape(-1)
    if array.size == 1:
        return np.full(n, float(array[0]), dtype=float)
    if array.size < n:
        array = np.pad(array, (0, n - array.size), constant_values=fill)
    return array[:n]


def _calibration_item(calibration: Any, kind: str, *args: int) -> Any:
    if calibration is None:
        return None
    method = getattr(calibration, kind, None)
    if callable(method):
        try:
            return method(*args)
        except (IndexError, TypeError):
            # 3Q group lookup does not use cycle, and cycle-residual may be absent.
            try:
                return method(*args[:2])
            except (IndexError, TypeError):
                return None
    return None


def _value(item: Any, name: str, neutral: Any) -> tuple[Any, bool]:
    if item is None or getattr(item, name, None) is None:
        return neutral, True
    return getattr(item, name), False


def _read_projection_sign(point: OperationPointSpec, channels: list[str]) -> float:
    signs = [float(point.diffraction_order_signs.get(channel, 1.0)) for channel in channels if channel.startswith("R_")]
    return float(np.mean(signs)) if signs else 1.0


def _total_power(point: OperationPointSpec, index: int) -> float:
    powers = []
    for command in point.channel_commands.values():
        amplitudes = command.get("amplitudes") or []
        powers.extend(float(value) ** 2 for value in amplitudes if value is not None)
    return float(sum(powers))


def _uncertainty(point: OperationPointSpec, calibration: Any, variance_scale: Any) -> Dict[str, Any]:
    phase_uncertainty = []
    confidence = []
    for row, col in point.cell_coordinates:
        cell = _calibration_item(calibration, "cell", row, col)
        phase_uncertainty.append(_value(cell, "phase_uncertainty_rad", None)[0])
        confidence.append(_value(cell, "confidence", 0.0)[0])
    known_phase = [value for value in phase_uncertainty if value is not None]
    return {
        "phase_uncertainty_rad_mean": float(np.mean(known_phase)) if known_phase else None,
        "calibration_confidence_mean": float(np.mean(confidence)) if confidence else 0.0,
        "variance_scale": None if variance_scale is None else float(variance_scale),
    }
