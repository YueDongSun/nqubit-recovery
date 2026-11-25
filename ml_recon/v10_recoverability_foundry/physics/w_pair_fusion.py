"""Bounded synthetic one-port fusion of two stored one-excitation resources.

This is a pure linear-optics model conditional on both inputs already being
stored, survived, and retrieved.  It deliberately contains no storage or
retrieval efficiency, calibration, dark-count, or generic-N experimental-map
claim.  The finite-efficiency detector model is synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from numbers import Real

import numpy as np


class Detector_mode(str, Enum):
    """The monitored-port detector event retained by this synthetic kernel."""

    THRESHOLD = "THRESHOLD"
    EXACT_ONE_NUMBER_RESOLVED = "EXACT_ONE_NUMBER_RESOLVED"


class W_pair_fusion_outcome(str, Enum):
    """A joint branch sampled from one supplied uniform variate."""

    ONE_EXCITATION_CLICK = "ONE_EXCITATION_CLICK"
    VACUUM_CLICK = "VACUUM_CLICK"
    NO_CLICK = "NO_CLICK"


@dataclass(frozen=True)
class W_pair_fusion_config:
    """Fixed synthetic detector and phase settings for one selected-mode pair."""

    selected_a: int
    selected_b: int
    detector_efficiency: float
    detector_mode: Detector_mode
    dephasing_visibility: float = 1.0
    monitored_port_phase_correction: complex = 1.0 + 0.0j
    source_tag: str = "SYNTHETIC"

    def __post_init__(self) -> None:
        if type(self.selected_a) is not int or self.selected_a < 0:
            raise ValueError("selected_a must be a nonnegative integer")
        if type(self.selected_b) is not int or self.selected_b < 0:
            raise ValueError("selected_b must be a nonnegative integer")
        if not isinstance(self.detector_mode, Detector_mode):
            raise ValueError("detector_mode must be a Detector_mode")
        _validate_probability(self.detector_efficiency, "detector_efficiency")
        _validate_probability(self.dephasing_visibility, "dephasing_visibility")
        phase = _finite_complex(
            self.monitored_port_phase_correction, "phase correction"
        )
        if not np.isclose(abs(phase), 1.0, atol=1.0e-12, rtol=0.0):
            raise ValueError("monitored_port_phase_correction must have unit magnitude")
        if self.source_tag != "SYNTHETIC":
            raise ValueError(
                "this unparameterized kernel permits source_tag=SYNTHETIC only"
            )

    def canonical_dict(self) -> dict[str, object]:
        """Return JSON-safe scalar settings bound into this synthetic model."""
        phase = complex(self.monitored_port_phase_correction)
        return {
            "schema": "W_pair_fusion_config_v1",
            "selected_a": self.selected_a,
            "selected_b": self.selected_b,
            "detector_efficiency": float(self.detector_efficiency),
            "detector_mode": self.detector_mode.value,
            "dephasing_visibility": float(self.dephasing_visibility),
            "monitored_port_phase_correction": [phase.real, phase.imag],
            "source_tag": self.source_tag,
        }

    def sha256(self) -> str:
        payload = json.dumps(
            self.canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class W_pair_fusion_result:
    """Exact weights and click-conditioned state for the synthetic kernel.

    ``conditional_one_excitation_block`` has trace
    ``true_one_excitation_click_weight / click_weight``.  Its remaining trace
    complement is ``conditional_vacuum_click_weight``; it is not a target
    certificate.  ``conditional_target_fidelity`` is the target overlap of
    that subnormalized click-conditioned block, so vacuum-click dilution is
    retained explicitly.
    """

    source_tag: str
    detector_mode: Detector_mode
    input_a_modes: int
    input_b_modes: int
    output_modes: int
    selected_a: int
    selected_b: int
    detector_efficiency: float
    dephasing_visibility: float
    monitored_port_phase_correction: complex
    true_one_excitation_click_weight: float
    vacuum_click_weight: float
    click_weight: float
    no_click_weight: float
    conditional_one_excitation_block: np.ndarray | None
    conditional_vacuum_click_weight: float | None
    conditional_target_fidelity: float | None


def evaluate_w_pair_fusion(
    input_a: np.ndarray,
    input_b: np.ndarray,
    config: W_pair_fusion_config,
    *,
    target: np.ndarray | None = None,
) -> W_pair_fusion_result:
    """Evaluate a one-port, two-resource W-pair fusion event without sampling.

    For normalized one-excitation inputs ``a`` and ``b``, the phase-corrected
    retained vector is ``u=(b_s a_rest, phase * a_s b_rest)/sqrt(2)``.  A
    real visibility only dephases the cross-resource coherences of its density
    block, which is a PSD channel rather than a success-probability factor.
    """
    if not isinstance(config, W_pair_fusion_config):
        raise ValueError("config must be a W_pair_fusion_config")
    a = _normalized_state(input_a, "input_a")
    b = _normalized_state(input_b, "input_b")
    if config.selected_a >= a.size or config.selected_b >= b.size:
        raise ValueError("selected mode is outside its input state")
    a_selected = a[config.selected_a]
    b_selected = b[config.selected_b]
    a_rest = np.delete(a, config.selected_a)
    b_rest = np.delete(b, config.selected_b)
    retained_vector = np.concatenate(
        (
            b_selected * a_rest,
            config.monitored_port_phase_correction * a_selected * b_rest,
        )
    ) / np.sqrt(2.0)
    rho_true = np.outer(retained_vector, retained_vector.conj())
    split = a_rest.size
    if split and b_rest.size:
        rho_true[:split, split:] *= config.dephasing_visibility
        rho_true[split:, :split] *= config.dephasing_visibility
    rho_true = 0.5 * (rho_true + rho_true.conj().T)
    true_weight = float(config.detector_efficiency * np.trace(rho_true).real)
    both_selected = float(abs(a_selected * b_selected) ** 2)
    vacuum_weight = _vacuum_click_weight(both_selected, config)
    click_weight = true_weight + vacuum_weight
    no_click_weight = _bounded_probability(1.0 - click_weight)
    block: np.ndarray | None = None
    vacuum_conditional: float | None = None
    fidelity: float | None = None
    if click_weight > 0.0:
        block = _freeze_array(config.detector_efficiency * rho_true / click_weight)
        vacuum_conditional = _bounded_probability(vacuum_weight / click_weight)
        if target is not None:
            target_vector = _normalized_target(target, block.shape[0])
            fidelity = _bounded_probability(
                float(np.vdot(target_vector, block @ target_vector).real)
            )
    elif target is not None:
        _normalized_target(target, a_rest.size + b_rest.size)
    return W_pair_fusion_result(
        source_tag=config.source_tag,
        detector_mode=config.detector_mode,
        input_a_modes=a.size,
        input_b_modes=b.size,
        output_modes=a_rest.size + b_rest.size,
        selected_a=config.selected_a,
        selected_b=config.selected_b,
        detector_efficiency=float(config.detector_efficiency),
        dephasing_visibility=float(config.dephasing_visibility),
        monitored_port_phase_correction=complex(config.monitored_port_phase_correction),
        true_one_excitation_click_weight=true_weight,
        vacuum_click_weight=vacuum_weight,
        click_weight=click_weight,
        no_click_weight=no_click_weight,
        conditional_one_excitation_block=block,
        conditional_vacuum_click_weight=vacuum_conditional,
        conditional_target_fidelity=fidelity,
    )


def sample_w_pair_fusion_outcome(
    result: W_pair_fusion_result, uniform: float
) -> W_pair_fusion_outcome:
    """Sample the physical branch with one joint uniform; never sample fidelity."""
    if not isinstance(result, W_pair_fusion_result):
        raise ValueError("result must be a W_pair_fusion_result")
    if (
        isinstance(uniform, bool)
        or not isinstance(uniform, Real)
        or not np.isfinite(uniform)
    ):
        raise ValueError("uniform must be a finite real value in [0,1)")
    if not 0.0 <= float(uniform) < 1.0:
        raise ValueError("uniform must lie in [0,1)")
    if uniform < result.true_one_excitation_click_weight:
        return W_pair_fusion_outcome.ONE_EXCITATION_CLICK
    if uniform < result.click_weight:
        return W_pair_fusion_outcome.VACUUM_CLICK
    return W_pair_fusion_outcome.NO_CLICK


def _vacuum_click_weight(both_selected: float, config: W_pair_fusion_config) -> float:
    eta = config.detector_efficiency
    if config.detector_mode is Detector_mode.THRESHOLD:
        return float(both_selected * (2.0 * eta - eta * eta) / 2.0)
    return float(both_selected * eta * (1.0 - eta))


def _normalized_state(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.size < 2 or array.dtype.kind not in "fciu":
        raise ValueError(
            f"{name} must be a one-dimensional numeric state with at least two modes"
        )
    vector = np.asarray(array, dtype=np.complex128)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    norm = float(np.vdot(vector, vector).real)
    if not np.isclose(norm, 1.0, atol=1.0e-12, rtol=1.0e-12):
        raise ValueError(f"{name} must already be normalized")
    return vector


def _normalized_target(value: np.ndarray, output_modes: int) -> np.ndarray:
    target = _normalized_state(value, "target")
    if target.size != output_modes:
        raise ValueError("target dimension must equal n_a+n_b-2")
    return target


def _validate_probability(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real probability")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must lie in [0,1]")


def _finite_complex(value: complex, name: str) -> complex:
    if isinstance(value, bool) or not isinstance(value, (Real, complex, np.number)):
        raise ValueError(f"{name} must be a finite complex scalar")
    result = complex(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _bounded_probability(value: float) -> float:
    if value < -1.0e-12 or value > 1.0 + 1.0e-12:
        raise RuntimeError("internal probability escaped [0,1]")
    return float(np.clip(value, 0.0, 1.0))


def _freeze_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.complex128)
    # A bytes-backed view cannot be made writable by changing NumPy flags.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
