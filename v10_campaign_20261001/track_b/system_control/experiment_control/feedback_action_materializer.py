"""Hardware projection for global W projective feedback actions.

This module prepares bounded candidate controls only.  It never writes the
shared matrix tree and never uploads an AWG waveform.  Read-side, brightness,
local, and cycle residual actions are rejected in the first hardware version.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from ml_recon.physics_hybrid_v5.projective_action import (
    amplitude_to_coordinates,
    coordinates_to_amplitude,
    coordinates_to_phase,
    phase_to_coordinates,
    wrap_phase,
)

from .data_protocol import stable_json_hash
from .models import ControlStateSpecV1, TargetStateSpecV1


@dataclass(frozen=True)
class HardwareProjectionLimits:
    step_amplitude_fraction: float = 0.005
    cumulative_amplitude_fraction: float = 0.03
    step_phase_rad: float = 0.02
    cumulative_phase_rad: float = 0.10
    emergency_amplitude_fraction: float = 0.03
    emergency_phase_rad: float = 0.20
    matrix_amplitude_min: float = 0.0
    matrix_amplitude_max: float = 4.0
    maximum_per_channel_rf_power: float = 4.0
    maximum_total_rf_power: float = 32.0
    maximum_waveform_peak: float = 1.0
    maximum_crest_factor: float = 12.0
    maximum_requested_actual_discrepancy: float = 1e-3


@dataclass(frozen=True)
class ActionMaterializationResult:
    requested_action: Mapping[str, Any]
    projected_action: Mapping[str, Any]
    materialized_action: Mapping[str, Any]
    actual_loaded_action: Mapping[str, Any]
    parent_control_state: ControlStateSpecV1
    requested_control_state: ControlStateSpecV1
    projected_control_state: ControlStateSpecV1
    actual_control_state: ControlStateSpecV1
    per_tone_delta_log_amp: Tuple[float, ...]
    per_tone_delta_phase: Tuple[float, ...]
    cumulative_delta_log_amp: Tuple[float, ...]
    cumulative_delta_phase: Tuple[float, ...]
    clipping_fraction: float
    power_report: Mapping[str, Any]
    fourier_report: Mapping[str, Any]
    accepted: bool
    reasons: Tuple[str, ...]

    @property
    def requested_action_hash(self) -> str:
        return stable_json_hash(dict(self.requested_action))

    @property
    def projected_action_hash(self) -> str:
        return stable_json_hash(dict(self.projected_action))

    @property
    def actual_action_hash(self) -> str:
        return stable_json_hash(dict(self.actual_loaded_action))

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "requested_action": dict(self.requested_action),
            "projected_action": dict(self.projected_action),
            "materialized_action": dict(self.materialized_action),
            "actual_loaded_action": dict(self.actual_loaded_action),
            "parent_control_state": self.parent_control_state.to_json_dict(),
            "requested_control_state": self.requested_control_state.to_json_dict(),
            "projected_control_state": self.projected_control_state.to_json_dict(),
            "actual_control_state": self.actual_control_state.to_json_dict(),
            "per_tone_delta_log_amp": list(self.per_tone_delta_log_amp),
            "per_tone_delta_phase": list(self.per_tone_delta_phase),
            "cumulative_delta_log_amp": list(self.cumulative_delta_log_amp),
            "cumulative_delta_phase": list(self.cumulative_delta_phase),
            "clipping_fraction": float(self.clipping_fraction),
            "power_report": dict(self.power_report),
            "fourier_report": dict(self.fourier_report),
            "accepted": bool(self.accepted),
            "reasons": list(self.reasons),
            "requested_action_hash": self.requested_action_hash,
            "projected_action_hash": self.projected_action_hash,
            "actual_action_hash": self.actual_action_hash,
        }


class FeedbackActionMaterializer:
    """Project global W amplitude/phase coordinates into engineering bounds."""

    def __init__(self, limits: HardwareProjectionLimits | None = None) -> None:
        self.limits = limits or HardwareProjectionLimits()

    def materialize(
        self,
        target_state: TargetStateSpecV1,
        control_state: ControlStateSpecV1,
        requested_action: Mapping[str, Any],
        *,
        waveform_diagnostics: Mapping[str, Any] | None = None,
        fourier_report: Mapping[str, Any] | None = None,
    ) -> ActionMaterializationResult:
        if target_state.system_type != control_state.system_type:
            raise ValueError("target/control system_type mismatch")
        cell_count = len(control_state.commanded_amplitudes)
        coordinate_count = cell_count - 1
        requested_amp = np.asarray(requested_action.get("amplitude_coordinates", np.zeros(coordinate_count)), dtype=float)
        requested_phase = np.asarray(requested_action.get("phase_coordinates", np.zeros(coordinate_count)), dtype=float)
        if requested_amp.shape != (coordinate_count,) or requested_phase.shape != (coordinate_count,):
            raise ValueError("global action coordinate dimension mismatch")
        reasons = self._forbidden_action_reasons(requested_action)
        if not np.all(np.isfinite(np.concatenate((requested_amp, requested_phase)))):
            raise ValueError("requested action contains non-finite values")

        current_amplitude = np.asarray(control_state.commanded_amplitudes, dtype=float)
        if np.any(current_amplitude <= 0.0) or float(np.linalg.norm(current_amplitude)) <= 0.0:
            raise ValueError("current control amplitudes must be strictly positive")
        current_amplitude /= np.linalg.norm(current_amplitude)
        current_phase = np.asarray(control_state.commanded_phases_rad, dtype=float)
        baseline_amplitude = coordinates_to_amplitude(
            amplitude_to_coordinates(current_amplitude)
            - np.asarray(control_state.cumulative_amplitude_coordinates, dtype=float)
        )
        baseline_phase = coordinates_to_phase(
            phase_to_coordinates(current_phase)
            - np.asarray(control_state.cumulative_phase_coordinates, dtype=float)
        )

        emergency_amp = float(np.log1p(self.limits.emergency_amplitude_fraction))
        emergency_phase = float(self.limits.emergency_phase_rad)
        requested_amplitude_unbounded = coordinates_to_amplitude(
            amplitude_to_coordinates(current_amplitude) + requested_amp
        )
        requested_phase_unbounded = wrap_phase(
            coordinates_to_phase(phase_to_coordinates(current_phase) + requested_phase)
        )
        hard_amp = np.clip(requested_amp, -emergency_amp, emergency_amp)
        hard_phase = np.clip(requested_phase, -emergency_phase, emergency_phase)
        requested_amplitude = coordinates_to_amplitude(
            amplitude_to_coordinates(current_amplitude) + hard_amp
        )
        requested_phase_values = wrap_phase(
            coordinates_to_phase(phase_to_coordinates(current_phase) + hard_phase)
        )

        step_log_limit = float(np.log1p(self.limits.step_amplitude_fraction))
        raw_step_log = np.log(np.maximum(requested_amplitude, 1e-12) / np.maximum(current_amplitude, 1e-12))
        step_log = np.clip(raw_step_log, -step_log_limit, step_log_limit)
        projected_amplitude = current_amplitude * np.exp(step_log)
        projected_amplitude /= np.linalg.norm(projected_amplitude)
        cumulative_log_limit = float(np.log1p(self.limits.cumulative_amplitude_fraction))
        cumulative_log = np.log(
            np.maximum(projected_amplitude, 1e-12) / np.maximum(baseline_amplitude, 1e-12)
        )
        bounded_cumulative_log = np.clip(cumulative_log, -cumulative_log_limit, cumulative_log_limit)
        projected_amplitude = baseline_amplitude * np.exp(bounded_cumulative_log)
        projected_amplitude /= np.linalg.norm(projected_amplitude)

        raw_step_phase = wrap_phase(requested_phase_values - current_phase)
        step_phase = np.clip(raw_step_phase, -self.limits.step_phase_rad, self.limits.step_phase_rad)
        projected_phase = wrap_phase(current_phase + step_phase)
        cumulative_phase = wrap_phase(projected_phase - baseline_phase)
        bounded_cumulative_phase = np.clip(
            cumulative_phase, -self.limits.cumulative_phase_rad, self.limits.cumulative_phase_rad,
        )
        projected_phase = wrap_phase(baseline_phase + bounded_cumulative_phase)
        projected_phase = wrap_phase(projected_phase - np.mean(projected_phase))

        projected_amp_coordinates = amplitude_to_coordinates(projected_amplitude) - amplitude_to_coordinates(current_amplitude)
        projected_phase_coordinates = phase_to_coordinates(projected_phase) - phase_to_coordinates(current_phase)
        cumulative_amp_coordinates = (
            amplitude_to_coordinates(projected_amplitude) - amplitude_to_coordinates(baseline_amplitude)
        )
        cumulative_phase_coordinates = (
            phase_to_coordinates(projected_phase) - phase_to_coordinates(baseline_phase)
        )
        projected_action = {
            "schema": "GlobalWProjectiveActionV1",
            "system_type": control_state.system_type,
            "amplitude_coordinates": projected_amp_coordinates.tolist(),
            "phase_coordinates": projected_phase_coordinates.tolist(),
            "brightness_delta": 0.0,
            "read_action": 0.0,
            "local_residual": 0.0,
            "cycle_residual": 0.0,
        }
        projected_control = control_state.evolve(
            commanded_amplitudes=projected_amplitude.tolist(),
            commanded_phases_rad=projected_phase.tolist(),
            cumulative_amplitude_coordinates=cumulative_amp_coordinates.tolist(),
            cumulative_phase_coordinates=cumulative_phase_coordinates.tolist(),
        )
        clip_flags = np.concatenate((
            np.abs(raw_step_log - step_log) > 1e-12,
            np.abs(cumulative_log - bounded_cumulative_log) > 1e-12,
            np.abs(raw_step_phase - step_phase) > 1e-12,
            np.abs(cumulative_phase - bounded_cumulative_phase) > 1e-12,
        ))
        clipping_fraction = float(np.mean(clip_flags))
        power_report = self._power_report(projected_amplitude, waveform_diagnostics or {})
        reasons.extend(power_report["reasons"])
        fourier = dict(fourier_report or {})
        if fourier and not bool(fourier.get("passed", not fourier.get("has_frequency_issues", True))):
            reasons.append("fourier_guard_failed")
        if np.any(projected_amplitude <= 0.0):
            reasons.append("non_positive_amplitude")
        if abs(float(np.linalg.norm(projected_amplitude)) - 1.0) > 1e-9:
            reasons.append("relative_amplitude_not_normalized")
        return ActionMaterializationResult(
            requested_action=self._json_action(requested_action),
            projected_action=projected_action,
            materialized_action=projected_action,
            actual_loaded_action=projected_action,
            parent_control_state=control_state,
            requested_control_state=control_state.evolve(
                commanded_amplitudes=requested_amplitude_unbounded.tolist(),
                commanded_phases_rad=requested_phase_unbounded.tolist(),
                cumulative_amplitude_coordinates=(
                    amplitude_to_coordinates(requested_amplitude_unbounded)
                    - amplitude_to_coordinates(baseline_amplitude)
                ).tolist(),
                cumulative_phase_coordinates=(
                    phase_to_coordinates(requested_phase_unbounded)
                    - phase_to_coordinates(baseline_phase)
                ).tolist(),
            ),
            projected_control_state=projected_control,
            actual_control_state=projected_control,
            per_tone_delta_log_amp=tuple(float(value) for value in np.log(projected_amplitude / current_amplitude)),
            per_tone_delta_phase=tuple(float(value) for value in wrap_phase(projected_phase - current_phase)),
            cumulative_delta_log_amp=tuple(
                float(value) for value in np.log(projected_amplitude / baseline_amplitude)
            ),
            cumulative_delta_phase=tuple(
                float(value) for value in wrap_phase(projected_phase - baseline_phase)
            ),
            clipping_fraction=clipping_fraction,
            power_report=power_report,
            fourier_report=fourier,
            accepted=not reasons,
            reasons=tuple(dict.fromkeys(reasons)),
        )

    def bind_actual_loaded(
        self,
        result: ActionMaterializationResult,
        target_state: TargetStateSpecV1,
        actual_control_state: ControlStateSpecV1,
        *,
        waveform_diagnostics: Mapping[str, Any],
        fourier_report: Mapping[str, Any],
    ) -> ActionMaterializationResult:
        current = result.parent_control_state
        actual_amplitude = np.asarray(actual_control_state.commanded_amplitudes, dtype=float)
        actual_phase = np.asarray(actual_control_state.commanded_phases_rad, dtype=float)
        current_amplitude = np.asarray(current.commanded_amplitudes, dtype=float)
        current_phase = np.asarray(current.commanded_phases_rad, dtype=float)
        actual_action = {
            "schema": "GlobalWProjectiveActionV1",
            "system_type": current.system_type,
            "amplitude_coordinates": (
                amplitude_to_coordinates(actual_amplitude) - amplitude_to_coordinates(current_amplitude)
            ).tolist(),
            "phase_coordinates": (
                phase_to_coordinates(actual_phase) - phase_to_coordinates(current_phase)
            ).tolist(),
            "brightness_delta": 0.0,
            "read_action": 0.0,
            "local_residual": 0.0,
            "cycle_residual": 0.0,
        }
        discrepancy = float(np.linalg.norm(
            np.asarray(actual_action["amplitude_coordinates"] + actual_action["phase_coordinates"])
            - np.asarray(
                list(result.projected_action["amplitude_coordinates"])
                + list(result.projected_action["phase_coordinates"])
            )
        ))
        power_report = self._power_report(actual_amplitude, waveform_diagnostics)
        power_report = {**power_report, "requested_actual_discrepancy": discrepancy}
        reasons = list(result.reasons)
        reasons.extend(power_report["reasons"])
        fourier = dict(fourier_report)
        if not bool(fourier.get("passed", not fourier.get("has_frequency_issues", True))):
            reasons.append("fourier_guard_failed")
        if discrepancy > self.limits.maximum_requested_actual_discrepancy:
            reasons.append("requested_actual_discrepancy_exceeded")
        return replace(
            result,
            actual_loaded_action=actual_action,
            actual_control_state=actual_control_state,
            power_report=power_report,
            fourier_report=fourier,
            accepted=not reasons,
            reasons=tuple(dict.fromkeys(reasons)),
        )

    def _power_report(self, amplitudes: np.ndarray, waveform: Mapping[str, Any]) -> Dict[str, Any]:
        per_channel_power = np.asarray(
            waveform.get("per_channel_rf_power", np.square(np.asarray(amplitudes, dtype=float))),
            dtype=float,
        )
        total_power = float(waveform.get("total_rf_power", np.sum(per_channel_power)))
        peak = float(waveform.get("waveform_peak", 0.0) or 0.0)
        crest = float(waveform.get("crest_factor", 0.0) or 0.0)
        clipping = float(waveform.get("clipping_fraction", 0.0) or 0.0)
        matrix_min = float(waveform.get("matrix_amplitude_min", np.min(amplitudes)))
        matrix_max = float(waveform.get("matrix_amplitude_max", np.max(amplitudes)))
        reasons = []
        if matrix_max > self.limits.matrix_amplitude_max or matrix_min < self.limits.matrix_amplitude_min:
            reasons.append("matrix_cell_amplitude_out_of_bounds")
        if float(np.max(per_channel_power)) > self.limits.maximum_per_channel_rf_power:
            reasons.append("per_channel_rf_power_exceeded")
        if total_power > self.limits.maximum_total_rf_power:
            reasons.append("total_rf_power_exceeded")
        if peak > self.limits.maximum_waveform_peak:
            reasons.append("waveform_peak_exceeded")
        if crest > self.limits.maximum_crest_factor:
            reasons.append("crest_factor_exceeded")
        if clipping > 0.0:
            reasons.append("waveform_clipping_detected")
        return {
            "per_channel_rf_power": per_channel_power.tolist(),
            "total_rf_power": total_power,
            "waveform_peak": peak,
            "crest_factor": crest,
            "clipping_fraction": clipping,
            "matrix_amplitude_min": matrix_min,
            "matrix_amplitude_max": matrix_max,
            "reasons": reasons,
        }

    @staticmethod
    def _forbidden_action_reasons(action: Mapping[str, Any]) -> list[str]:
        reasons = []
        for name in ("brightness_delta", "read_action", "local_residual", "cycle_residual"):
            value = action.get(name, 0.0)
            if np.any(np.asarray(value, dtype=float) != 0.0):
                reasons.append(f"forbidden_nonzero_{name}")
        return reasons

    @staticmethod
    def _json_action(action: Mapping[str, Any]) -> Dict[str, Any]:
        value = {}
        for name, item in action.items():
            value[str(name)] = np.asarray(item).tolist() if isinstance(item, (np.ndarray, tuple, list)) else item
        return value


__all__ = [
    "ActionMaterializationResult",
    "FeedbackActionMaterializer",
    "HardwareProjectionLimits",
]
