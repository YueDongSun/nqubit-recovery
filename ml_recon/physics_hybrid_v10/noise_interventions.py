"""Executable state/apparatus/detector interventions and actual D3 matching."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Tuple

import numpy as np

from ml_recon.data_v2_4.hashing import sha256_json
from ml_recon.data_v2_4.version import NOISE_INTERVENTION_SHA

from .dense_pauli_panel import Dense_Pauli_Panel, bind_dense_pauli_panel
from .measurement_model import born_probabilities
from .noise_matching import hellinger_squared, match_hellinger_severity, state_fidelity
from .state_backends import DenseDensityStateBackend


CORRECTABLE_INTERVENTIONS = (
    "finite_amplitude_error",
    "finite_relative_phase_error",
    "aod_nonlinear_amplitude_response",
    "aod_phase_response",
    "amplitude_phase_coupling",
    "calibration_drift_control_subspace",
)

IRRECOVERABLE_INTERVENTIONS = (
    "correlated_dephasing",
    "w_subspace_leakage",
    "explicit_multi_excitation",
    "detector_dark",
    "detector_background",
    "detector_accidentals",
    "retrieval_efficiency_imbalance",
    "brightness_drift",
    "mechanism_shift",
)


def _measurement_backend(
    state: np.ndarray,
    settings: Tuple[Tuple[str, ...], ...],
    panel: Dense_Pauli_Panel | None,
):
    backend = DenseDensityStateBackend(state)
    wrapped_backend = bind_dense_pauli_panel(backend, panel)
    if panel is not None:
        panel.validate_settings(settings)
    return wrapped_backend


def _normalized_density(rho: np.ndarray) -> np.ndarray:
    value = np.asarray(rho, dtype=np.complex128)
    value = 0.5 * (value + value.conj().T)
    trace = float(np.trace(value).real)
    if trace <= 0.0:
        raise ValueError("noise channel produced zero density trace")
    value /= trace
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    if eigenvalues.min(initial=0.0) < -1e-9:
        raise ValueError("noise channel produced a non-PSD density matrix")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    value = (eigenvectors * eigenvalues) @ eigenvectors.conj().T
    return value / np.trace(value).real


def _basis_bits(N: int) -> np.ndarray:
    indices = np.arange(2 ** N, dtype=np.uint64)
    shifts = np.arange(N - 1, -1, -1, dtype=np.uint64)
    return ((indices[:, None] >> shifts[None, :]) & 1).astype(np.float64)


def _filter_density(rho: np.ndarray, diagonal: np.ndarray) -> np.ndarray:
    filtered = diagonal[:, None] * np.asarray(rho) * diagonal.conj()[None, :]
    return _normalized_density(filtered)


def _unitary_phase_density(rho: np.ndarray, phase: np.ndarray) -> np.ndarray:
    diagonal = np.exp(1j * phase)
    return _normalized_density(diagonal[:, None] * rho * diagonal.conj()[None, :])


@dataclass(frozen=True)
class V24NoiseIntervention:
    mechanism_name: str
    category: str
    channel_scope: str
    maximum_parameter: float
    expected_statistic: str

    def metadata(self) -> Dict[str, Any]:
        return {
            "mechanism_name": self.mechanism_name,
            "category": self.category,
            "channel_scope": self.channel_scope,
            "maximum_parameter": self.maximum_parameter,
            "expected_statistic": self.expected_statistic,
            "intervention_forward_sha": NOISE_INTERVENTION_SHA,
        }

    def physical_parameter(self, severity: float) -> Dict[str, float | str]:
        if not 0.0 <= severity <= self.maximum_parameter:
            raise ValueError("noise severity is outside its registered physical bracket")
        return {"name": self.mechanism_name, "value": float(severity)}

    def apply_state_channel(self, rho: np.ndarray, severity: float) -> np.ndarray:
        value = _normalized_density(rho)
        N = int(round(np.log2(value.shape[0])))
        bits = _basis_bits(N)
        position = bits @ np.linspace(-1.0, 1.0, N)
        name = self.mechanism_name
        if name == "finite_amplitude_error":
            return _filter_density(value, 1.0 + severity * position)
        if name == "finite_relative_phase_error":
            return _unitary_phase_density(value, severity * np.pi * position)
        if name == "aod_nonlinear_amplitude_response":
            return _filter_density(value, np.exp(severity * position**2))
        if name == "aod_phase_response":
            return _unitary_phase_density(value, severity * np.pi * np.sin(position))
        if name == "amplitude_phase_coupling":
            filtered = _filter_density(value, np.exp(0.5 * severity * position))
            return _unitary_phase_density(filtered, severity * np.pi * position**2)
        if name == "calibration_drift_control_subspace":
            filtered = _filter_density(value, 1.0 + 0.5 * severity * position)
            return _unitary_phase_density(filtered, severity * np.pi * position)
        if name in ("correlated_dephasing", "mechanism_shift"):
            hamming = np.abs(bits[:, None, :] - bits[None, :, :]).sum(-1)
            scale = severity * (1.5 if name == "mechanism_shift" else 1.0)
            value = _normalized_density(value * np.exp(-scale * hamming))
        if name == "w_subspace_leakage":
            vacuum = np.zeros_like(value)
            vacuum[0, 0] = 1.0
            value = _normalized_density((1.0 - severity) * value + severity * vacuum)
        if name == "explicit_multi_excitation":
            multi = np.zeros_like(value)
            multi[(1 << (N - 1)) | (1 << (N - 2)),
                  (1 << (N - 1)) | (1 << (N - 2))] = 1.0
            value = _normalized_density((1.0 - severity) * value + severity * multi)
        return value

    def apply_apparatus_channel(self, probabilities: np.ndarray,
                                outcomes: np.ndarray, severity: float) -> np.ndarray:
        probability = np.asarray(probabilities, dtype=np.float64)
        if self.mechanism_name in ("finite_amplitude_error", "aod_nonlinear_amplitude_response"):
            detected = (np.asarray(outcomes) < 0).astype(np.float64)
            imbalance = detected @ np.linspace(-1.0, 1.0, detected.shape[1])
            probability = probability * np.exp(severity * imbalance)
        if self.mechanism_name == "retrieval_efficiency_imbalance":
            detected = (np.asarray(outcomes) < 0).astype(np.float64)
            imbalance = detected @ np.linspace(-1.0, 1.0, detected.shape[1])
            probability = probability * np.exp(severity * imbalance)
        probability = np.maximum(probability, 0.0)
        return probability / max(probability.sum(), 1e-15)

    def apply_detector_channel(self, probabilities: np.ndarray,
                               severity: float) -> np.ndarray:
        probability = np.asarray(probabilities, dtype=np.float64)
        name = self.mechanism_name
        detection = 0.95
        if name == "detector_dark":
            probability = (1.0 - severity) * probability \
                + severity * np.ones_like(probability) / probability.size
            detection = min(1.0, detection + 0.04 * severity)
        elif name == "detector_background":
            background = np.linspace(1.0, 2.0, probability.size)
            background /= background.sum()
            probability = (1.0 - severity) * probability + severity * background
        elif name == "detector_accidentals":
            accidental = np.roll(probability, 1)
            probability = (1.0 - severity) * probability + severity * accidental
        elif name == "brightness_drift":
            detection = np.clip(detection * (1.0 - 0.9 * severity), 0.05, 1.0)
        elif name == "mechanism_shift":
            probability = (1.0 - severity) * probability \
                + severity * np.ones_like(probability) / probability.size
            detection = np.clip(detection * (1.0 - 0.5 * severity), 0.05, 1.0)
        probability = np.maximum(probability, 0.0)
        probability /= max(probability.sum(), 1e-15)
        return np.concatenate((detection * probability, np.asarray((1.0 - detection,))))

    def measurement_law(self, rho: np.ndarray,
                        settings: Tuple[Tuple[str, ...], ...],
                        severity: float,
                        *,
                        panel: Dense_Pauli_Panel | None = None) -> np.ndarray:
        state = self.apply_state_channel(rho, severity)
        backend = _measurement_backend(state, settings, panel)
        rows = []
        for axes in settings:
            outcomes, probability = born_probabilities(backend, axes)
            probability = self.apply_apparatus_channel(probability, outcomes, severity)
            rows.append(self.apply_detector_channel(probability, severity))
        flattened = np.concatenate(rows)
        return flattened / flattened.sum()

    def severity_matching(self, rho: np.ndarray,
                          settings: Tuple[Tuple[str, ...], ...],
                          eta_target: float,
                          *,
                          panel: Dense_Pauli_Panel | None = None) -> Dict[str, Any]:
        identity = V24NoiseIntervention(
            "identity", "none", "none", 0.0, "none")
        clean = identity.measurement_law(rho, settings, 0.0, panel=panel)
        maximum_law = self.measurement_law(
            rho,
            settings,
            self.maximum_parameter,
            panel=panel,
        )
        maximum_eta = hellinger_squared(clean, maximum_law)
        if not 0.0 <= eta_target <= maximum_eta + 1e-10:
            raise ValueError(
                f"eta_target {eta_target} exceeds mechanism maximum {maximum_eta}")
        matched = match_hellinger_severity(
            clean,
            lambda value: self.measurement_law(
                rho,
                settings,
                value,
                panel=panel,
            ),
            eta_target, lower=0.0, upper=self.maximum_parameter,
            tolerance=1e-7, iterations=24)
        parameter = float(matched["physical_channel_parameter"])
        achieved = float(matched["eta_Hellinger"])
        noisy_rho = self.apply_state_channel(rho, parameter)
        residual = abs(achieved - eta_target)
        calibration = d3_heldout_calibration(self.mechanism_name, len(settings[0]))
        return {
            "eta_target": float(eta_target),
            "eta_achieved": achieved,
            "matching_residual": residual,
            "converged": residual <= 1e-6,
            "physical_parameter": parameter,
            "measurement_clean": clean,
            "measurement_noisy": np.asarray(matched["measurement_noisy"]),
            "rho_noisy": noisy_rho,
            "state_infidelity": 1.0 - state_fidelity(rho, noisy_rho),
            "train_fit_sha": calibration["train_fit_sha"],
            "heldout_error": calibration["heldout_error"],
            "intervention_forward_sha": NOISE_INTERVENTION_SHA,
        }


@dataclass(frozen=True)
class LockedOODIntervention:
    """Held-out correlated phase/apparatus/detector law for locked OOD."""

    mechanism_name: str = "locked_spatial_phase_detector_shift"
    category: str = "locked_ood"
    physical_parameter: float = 0.72

    def apply_state_channel(self, rho: np.ndarray, severity: float | None = None) -> np.ndarray:
        value = _normalized_density(rho)
        N = int(round(np.log2(value.shape[0])))
        bits = _basis_bits(N)
        position = bits @ np.linspace(-1.0, 1.0, N)
        strength = self.physical_parameter if severity is None else float(severity)
        phase = strength * (
            0.55 * np.sin(2.0 * np.pi * position)
            + 0.35 * np.sin(4.0 * np.pi * position + 0.7))
        return _unitary_phase_density(value, phase)

    def apply_apparatus_channel(self, probabilities: np.ndarray,
                                outcomes: np.ndarray,
                                severity: float | None = None) -> np.ndarray:
        strength = self.physical_parameter if severity is None else float(severity)
        parity = np.prod(np.asarray(outcomes), axis=1).astype(np.float64)
        value = np.asarray(probabilities, dtype=np.float64) * np.exp(
            0.8 * strength * parity)
        return value / max(float(value.sum()), 1e-15)

    def apply_detector_channel(self, probabilities: np.ndarray,
                               severity: float | None = None) -> np.ndarray:
        strength = self.physical_parameter if severity is None else float(severity)
        value = np.asarray(probabilities, dtype=np.float64)
        ramp = np.linspace(0.5, 1.5, value.size)
        shifted = (1.0 - strength) * value + strength * ramp / ramp.sum()
        detection = float(np.clip(0.95 - 0.30 * strength, 0.05, 1.0))
        return np.concatenate((detection * shifted, np.asarray((1.0 - detection,))))

    def measurement_law(self, rho: np.ndarray,
                        settings: Tuple[Tuple[str, ...], ...],
                        severity: float | None = None,
                        *,
                        panel: Dense_Pauli_Panel | None = None) -> np.ndarray:
        state = self.apply_state_channel(rho, severity)
        backend = _measurement_backend(state, settings, panel)
        rows = []
        for axes in settings:
            outcomes, probability = born_probabilities(backend, axes)
            probability = self.apply_apparatus_channel(probability, outcomes, severity)
            rows.append(self.apply_detector_channel(probability, severity))
        flattened = np.concatenate(rows)
        return flattened / max(float(flattened.sum()), 1e-15)

@dataclass(frozen=True)
class CompositeNoiseIntervention:
    """Ordered correctable-then-irrecoverable physical channel."""

    correctable: V24NoiseIntervention
    irrecoverable: V24NoiseIntervention
    correctable_parameter: float
    irrecoverable_parameter: float

    @property
    def mechanism_name(self) -> str:
        return (f"composite:{self.correctable.mechanism_name}+"
                f"{self.irrecoverable.mechanism_name}")

    @property
    def category(self) -> str:
        return "composite"

    @property
    def physical_parameter(self) -> float:
        return float(np.hypot(self.correctable_parameter,
                              self.irrecoverable_parameter))

    def apply_state_channel(self, rho: np.ndarray, _severity: float = 0.0) -> np.ndarray:
        corrected = self.correctable.apply_state_channel(
            rho, self.correctable_parameter)
        return self.irrecoverable.apply_state_channel(
            corrected, self.irrecoverable_parameter)

    def apply_apparatus_channel(self, probabilities: np.ndarray,
                                outcomes: np.ndarray,
                                _severity: float = 0.0) -> np.ndarray:
        corrected = self.correctable.apply_apparatus_channel(
            probabilities, outcomes, self.correctable_parameter)
        return self.irrecoverable.apply_apparatus_channel(
            corrected, outcomes, self.irrecoverable_parameter)

    def apply_detector_channel(self, probabilities: np.ndarray,
                               _severity: float = 0.0) -> np.ndarray:
        corrected = self.correctable.apply_detector_channel(
            probabilities, self.correctable_parameter)
        return self.irrecoverable.apply_detector_channel(
            corrected[:-1], self.irrecoverable_parameter)

    def measurement_law(self, rho: np.ndarray,
                        settings: Tuple[Tuple[str, ...], ...],
                        _severity: float = 0.0,
                        *,
                        panel: Dense_Pauli_Panel | None = None) -> np.ndarray:
        state = self.apply_state_channel(rho)
        backend = _measurement_backend(state, settings, panel)
        rows = []
        for axes in settings:
            outcomes, probability = born_probabilities(backend, axes)
            probability = self.apply_apparatus_channel(probability, outcomes)
            rows.append(self.apply_detector_channel(probability))
        flattened = np.concatenate(rows)
        return flattened / max(float(flattened.sum()), 1e-15)


def intervention_registry() -> Dict[str, V24NoiseIntervention]:
    correctable_scope = {
        "finite_amplitude_error": "state_amplitude",
        "finite_relative_phase_error": "state_phase",
        "aod_nonlinear_amplitude_response": "apparatus_amplitude",
        "aod_phase_response": "apparatus_phase",
        "amplitude_phase_coupling": "apparatus_coupled",
        "calibration_drift_control_subspace": "state_and_apparatus",
    }
    irrecoverable_scope = {
        "correlated_dephasing": "state_coherence",
        "w_subspace_leakage": "state_sector",
        "explicit_multi_excitation": "state_sector",
        "detector_dark": "detector_rate",
        "detector_background": "detector_distribution",
        "detector_accidentals": "detector_coincidence",
        "retrieval_efficiency_imbalance": "apparatus_efficiency",
        "brightness_drift": "detector_absolute_rate",
        "mechanism_shift": "state_apparatus_detector",
    }
    registry = {
        name: V24NoiseIntervention(
            name, "correctable", correctable_scope[name], 0.8,
            "relative_amplitude_or_phase_response")
        for name in CORRECTABLE_INTERVENTIONS
    }
    registry.update({
        name: V24NoiseIntervention(
            name, "irrecoverable", irrecoverable_scope[name], 0.8,
            "coherence_sector_or_detector_rate")
        for name in IRRECOVERABLE_INTERVENTIONS
    })
    return registry


@lru_cache(maxsize=None)
def d3_heldout_calibration(mechanism_name: str, N: int) -> Dict[str, Any]:
    intervention = intervention_registry()[mechanism_name]
    settings = registered_feedback_settings(N)
    residuals = []
    train_ids = []
    heldout_ids = []
    for panel_name, seeds in (("train", (11, 17)), ("heldout", (23, 29))):
        for seed in seeds:
            rng = np.random.default_rng(seed + 1000 * N)
            coefficients = rng.normal(size=N) + 1j * rng.normal(size=N)
            coefficients /= np.linalg.norm(coefficients)
            rho = w_density_matrix(coefficients)
            clean = V24NoiseIntervention(
                "identity", "none", "none", 0.0, "none").measurement_law(
                    rho, settings, 0.0)
            maximum_eta = hellinger_squared(
                clean, intervention.measurement_law(
                    rho, settings, intervention.maximum_parameter))
            target = 0.35 * maximum_eta
            matched = match_hellinger_severity(
                clean, lambda value: intervention.measurement_law(rho, settings, value),
                target, lower=0.0, upper=intervention.maximum_parameter,
                tolerance=1e-6, iterations=16)
            residual = abs(float(matched["eta_Hellinger"]) - target)
            if panel_name == "heldout":
                residuals.append(residual)
                heldout_ids.append(f"d3:{mechanism_name}:N{N}:heldout:{seed}")
            else:
                train_ids.append(f"d3:{mechanism_name}:N{N}:train:{seed}")
    fit_payload = {
        "mechanism_name": mechanism_name, "N": N,
        "intervention_forward_sha": NOISE_INTERVENTION_SHA,
        "train_sample_ids": train_ids, "heldout_sample_ids": heldout_ids,
        "matching": "per_sample_bisection_v1",
    }
    return {
        "train_fit_sha": sha256_json(fit_payload),
        "heldout_error": float(np.mean(residuals)),
    }


def w_density_matrix(coefficients: np.ndarray) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.complex128)
    values /= np.linalg.norm(values)
    N = len(values)
    ket = np.zeros(2 ** N, dtype=np.complex128)
    for qubit, coefficient in enumerate(values):
        ket[1 << (N - 1 - qubit)] = coefficient
    return np.outer(ket, ket.conj())


def registered_feedback_settings(N: int) -> Tuple[Tuple[str, ...], ...]:
    settings = [tuple("Z" for _ in range(N))]
    for mode in range(1, N):
        xx = ["Z"] * N
        xy = ["Z"] * N
        xx[0] = xx[mode] = "X"
        xy[0], xy[mode] = "X", "Y"
        settings.extend((tuple(xx), tuple(xy)))
    return tuple(settings)
