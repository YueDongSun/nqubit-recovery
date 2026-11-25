"""Registry-faithful, count-regenerating 3Q/4Q V5 digital-twin core.

This module is simulation-only. It has no AWG, FPGA, runner, or delta-writing
imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from ml_recon.physics_hybrid_v4.phase_geometry import helmert_matrix
from system_control.experiment_control.operation_points import (
    build_operation_registry,
    operation_registry_hash,
)


COUNT_CHANNELS = ("signal", "idler", "coin", "multi2", "multi3", "multi4")


@dataclass(frozen=True)
class RegisteredTwinParameters:
    signal_rate_hz: float = 0.9
    idler_rate_hz: float = 0.85
    coin_rate_hz: float = 0.12
    dark_rate_hz: float = 1e-4
    nb2_overdispersion: float = 0.05
    operation_crosstalk: float = 0.0
    multi2_scale: float = 0.02
    multi3_scale: float = 0.002
    multi4_scale: float = 0.0002

    def validate(self) -> None:
        for name in (
            "signal_rate_hz",
            "idler_rate_hz",
            "coin_rate_hz",
            "dark_rate_hz",
            "multi2_scale",
            "multi3_scale",
            "multi4_scale",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if float(self.nb2_overdispersion) < 0.0:
            raise ValueError("nb2_overdispersion must be non-negative")
        if not 0.0 <= float(self.operation_crosstalk) <= 0.25:
            raise ValueError("operation_crosstalk must be in [0, 0.25]")


@dataclass(frozen=True)
class TwinObservation:
    exposure_s: float
    expected_counts: np.ndarray
    sampled_counts: np.ndarray
    reconstructed_rho_raw: np.ndarray
    reconstructed_rho_physical: np.ndarray
    physical_projection_distance: float


class RegisteredCountDigitalTwin:
    """Exact-basis count generator with differentiable phase coordinates."""

    def __init__(self, system_type: str):
        if system_type not in {"3qubit", "4qubit"}:
            raise ValueError("RegisteredCountDigitalTwin supports only 3qubit/4qubit")
        self.system_type = system_type
        self.registry = build_operation_registry(
            system_type,
            include_ml_delta=False,
        )
        self.registry_hash = operation_registry_hash(self.registry)
        self.amplitudes = np.asarray(
            [entry.basis_amplitudes for entry in self.registry],
            dtype=np.float64,
        )
        self.phases = np.asarray(
            [entry.basis_phases_rad for entry in self.registry],
            dtype=np.float64,
        )
        self.operation_count, self.dimension = self.amplitudes.shape
        expected = 75 if system_type == "3qubit" else 256
        if self.operation_count != expected:
            raise ValueError(
                f"{system_type} registry has {self.operation_count} operations, expected {expected}"
            )
        norms = np.linalg.norm(self.amplitudes, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError("operation registry contains an all-zero measurement amplitude")
        self.normalized_amplitudes = self.amplitudes / norms
        self.helmert = helmert_matrix(self.dimension).astype(np.float64)

    @property
    def coordinate_count(self) -> int:
        return self.dimension - 1

    def validate_density(self, density: np.ndarray, *, tolerance: float = 1e-8) -> None:
        rho = np.asarray(density, dtype=np.complex128)
        if rho.shape != (self.dimension, self.dimension):
            raise ValueError(
                f"density must have shape {(self.dimension, self.dimension)}"
            )
        if not np.all(np.isfinite(rho)):
            raise ValueError("density must be finite")
        if np.linalg.norm(rho - rho.conj().T) > tolerance:
            raise ValueError("density is not Hermitian within tolerance")
        if abs(float(np.real(np.trace(rho))) - 1.0) > tolerance:
            raise ValueError("density trace is not one within tolerance")
        if float(np.min(np.linalg.eigvalsh(rho))) < -tolerance:
            raise ValueError("density is not positive semidefinite within tolerance")

    def measurement_vectors(self, phase_coordinates: np.ndarray) -> np.ndarray:
        coordinates = self._coordinates(phase_coordinates)
        offsets = coordinates @ self.helmert.T
        return self.normalized_amplitudes * np.exp(1j * (self.phases + offsets))

    def born_probabilities(
        self,
        density: np.ndarray,
        phase_coordinates: np.ndarray,
    ) -> np.ndarray:
        self.validate_density(density)
        vectors = self.measurement_vectors(phase_coordinates)
        probability = np.real(
            np.einsum(
                "oi,ij,oj->o",
                np.conj(vectors),
                np.asarray(density, dtype=np.complex128),
                vectors,
            )
        )
        return np.clip(probability, 0.0, 1.0)

    def expected_rates(
        self,
        density: np.ndarray,
        phase_coordinates: np.ndarray,
        parameters: RegisteredTwinParameters,
    ) -> np.ndarray:
        parameters.validate()
        probability = self.born_probabilities(density, phase_coordinates)
        signal = np.full(self.operation_count, parameters.signal_rate_hz)
        idler = np.full(self.operation_count, parameters.idler_rate_hz)
        raw_coin = parameters.coin_rate_hz * probability
        coin = self._mix_operation_crosstalk(
            raw_coin,
            parameters.operation_crosstalk,
        )
        dark = float(parameters.dark_rate_hz)
        signal = signal + dark
        idler = idler + dark
        coin = coin + dark
        multi2 = parameters.multi2_scale * np.square(coin)
        multi3 = parameters.multi3_scale * np.power(coin, 3)
        multi4 = parameters.multi4_scale * np.power(coin, 4)
        return np.stack((signal, idler, coin, multi2, multi3, multi4), axis=-1)

    def expected_counts(
        self,
        density: np.ndarray,
        phase_coordinates: np.ndarray,
        parameters: RegisteredTwinParameters,
        *,
        exposure_s: float,
    ) -> np.ndarray:
        exposure = float(exposure_s)
        if exposure <= 0.0:
            raise ValueError("exposure_s must be positive")
        return self.expected_rates(density, phase_coordinates, parameters) * exposure

    def sample_counts(
        self,
        density: np.ndarray,
        phase_coordinates: np.ndarray,
        parameters: RegisteredTwinParameters,
        *,
        exposure_s: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        mean = self.expected_counts(
            density,
            phase_coordinates,
            parameters,
            exposure_s=exposure_s,
        )
        alpha = float(parameters.nb2_overdispersion)
        if alpha <= 1e-10:
            return rng.poisson(mean).astype(np.float64)
        inverse_dispersion = 1.0 / alpha
        probability = inverse_dispersion / (inverse_dispersion + mean)
        return rng.negative_binomial(inverse_dispersion, probability).astype(np.float64)

    def count_jacobian_analytic(
        self,
        density: np.ndarray,
        phase_coordinates: np.ndarray,
        parameters: RegisteredTwinParameters,
        *,
        exposure_s: float,
    ) -> np.ndarray:
        """Return d expected_counts[o,c] / d coordinates[j,k]."""

        self.validate_density(density)
        parameters.validate()
        coordinates = self._coordinates(phase_coordinates)
        vectors = self.measurement_vectors(coordinates)
        rho_v = np.einsum("ij,oj->oi", density, vectors)
        local_probability_jacobian = np.empty(
            (self.operation_count, self.coordinate_count),
            dtype=np.float64,
        )
        for coordinate in range(self.coordinate_count):
            derivative_vector = 1j * vectors * self.helmert[:, coordinate][None, :]
            local_probability_jacobian[:, coordinate] = 2.0 * np.real(
                np.einsum(
                    "oi,oi->o",
                    np.conj(derivative_vector),
                    rho_v,
                )
            )
        raw_coin_derivative = parameters.coin_rate_hz * local_probability_jacobian
        coin_derivative = self._crosstalk_jacobian(
            raw_coin_derivative,
            parameters.operation_crosstalk,
        )
        rates = self.expected_rates(density, coordinates, parameters)
        coin = rates[:, 2]
        jacobian = np.zeros(
            (
                self.operation_count,
                len(COUNT_CHANNELS),
                self.operation_count,
                self.coordinate_count,
            ),
            dtype=np.float64,
        )
        jacobian[:, 2] = coin_derivative
        jacobian[:, 3] = (
            2.0 * parameters.multi2_scale * coin[:, None, None] * coin_derivative
        )
        jacobian[:, 4] = (
            3.0
            * parameters.multi3_scale
            * np.square(coin)[:, None, None]
            * coin_derivative
        )
        jacobian[:, 5] = (
            4.0
            * parameters.multi4_scale
            * np.power(coin, 3)[:, None, None]
            * coin_derivative
        )
        return jacobian * float(exposure_s)

    def count_jacobian_central_difference(
        self,
        density: np.ndarray,
        phase_coordinates: np.ndarray,
        parameters: RegisteredTwinParameters,
        *,
        exposure_s: float,
        step_rad: float = 1e-6,
    ) -> np.ndarray:
        coordinates = self._coordinates(phase_coordinates)
        jacobian = np.empty(
            (
                self.operation_count,
                len(COUNT_CHANNELS),
                self.operation_count,
                self.coordinate_count,
            ),
            dtype=np.float64,
        )
        for operation in range(self.operation_count):
            for coordinate in range(self.coordinate_count):
                plus = coordinates.copy()
                minus = coordinates.copy()
                plus[operation, coordinate] += step_rad
                minus[operation, coordinate] -= step_rad
                jacobian[:, :, operation, coordinate] = (
                    self.expected_counts(
                        density, plus, parameters, exposure_s=exposure_s
                    )
                    - self.expected_counts(
                        density, minus, parameters, exposure_s=exposure_s
                    )
                ) / (2.0 * step_rad)
        return jacobian

    def count_jacobian_autograd(
        self,
        density: np.ndarray,
        phase_coordinates: np.ndarray,
        parameters: RegisteredTwinParameters,
        *,
        exposure_s: float,
    ) -> np.ndarray:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyTorch is required for the autograd Jacobian") from exc
        self.validate_density(density)
        parameters.validate()
        coordinates = torch.as_tensor(
            self._coordinates(phase_coordinates),
            dtype=torch.float64,
        ).requires_grad_(True)
        amplitude = torch.as_tensor(self.normalized_amplitudes, dtype=torch.float64)
        phase = torch.as_tensor(self.phases, dtype=torch.float64)
        helmert = torch.as_tensor(self.helmert, dtype=torch.float64)
        rho = torch.as_tensor(np.asarray(density), dtype=torch.complex128)

        def generator(value):
            offsets = value @ helmert.T
            vectors = amplitude.to(torch.complex128) * torch.exp(
                1j * (phase + offsets)
            )
            probability = torch.real(
                torch.einsum("oi,ij,oj->o", torch.conj(vectors), rho, vectors)
            )
            raw_coin = float(parameters.coin_rate_hz) * probability
            coupling = float(parameters.operation_crosstalk)
            coin = (
                (1.0 - coupling) * raw_coin
                + 0.5 * coupling * (torch.roll(raw_coin, 1) + torch.roll(raw_coin, -1))
                + float(parameters.dark_rate_hz)
            )
            signal = torch.full_like(coin, float(parameters.signal_rate_hz + parameters.dark_rate_hz))
            idler = torch.full_like(coin, float(parameters.idler_rate_hz + parameters.dark_rate_hz))
            return float(exposure_s) * torch.stack(
                (
                    signal,
                    idler,
                    coin,
                    float(parameters.multi2_scale) * coin.square(),
                    float(parameters.multi3_scale) * coin.pow(3),
                    float(parameters.multi4_scale) * coin.pow(4),
                ),
                dim=-1,
            )

        jacobian = torch.autograd.functional.jacobian(
            generator,
            coordinates,
            vectorize=True,
        )
        return jacobian.detach().cpu().numpy()

    def reconstruct_density(
        self,
        counts: np.ndarray,
        phase_coordinates: np.ndarray,
        parameters: RegisteredTwinParameters,
        *,
        exposure_s: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        values = np.asarray(counts, dtype=np.float64)
        if values.shape != (self.operation_count, len(COUNT_CHANNELS)):
            raise ValueError(
                f"counts must have shape {(self.operation_count, len(COUNT_CHANNELS))}"
            )
        coin_rate = values[:, 2] / float(exposure_s) - float(parameters.dark_rate_hz)
        raw_coin = self._invert_operation_crosstalk(
            coin_rate,
            parameters.operation_crosstalk,
        )
        probability = np.clip(
            raw_coin / max(float(parameters.coin_rate_hz), 1e-12),
            0.0,
            1.0,
        )
        vectors = self.measurement_vectors(phase_coordinates)
        operators = np.einsum("oi,oj->oij", np.conj(vectors), vectors)
        design = operators.reshape(self.operation_count, self.dimension * self.dimension)
        raw_vector, *_ = np.linalg.lstsq(design, probability, rcond=None)
        raw = raw_vector.reshape(self.dimension, self.dimension)
        raw = 0.5 * (raw + raw.conj().T)
        trace = float(np.real(np.trace(raw)))
        if abs(trace) > 1e-12:
            raw = raw / trace
        eigenvalues, eigenvectors = np.linalg.eigh(raw)
        clipped = np.maximum(eigenvalues, 0.0)
        if float(np.sum(clipped)) <= 1e-12:
            physical = np.eye(self.dimension, dtype=np.complex128) / self.dimension
        else:
            physical = (eigenvectors * clipped[None, :]) @ eigenvectors.conj().T
            physical /= np.trace(physical)
        distance = float(np.linalg.norm(physical - raw))
        return raw, physical, distance

    def observe(
        self,
        density: np.ndarray,
        phase_coordinates: np.ndarray,
        parameters: RegisteredTwinParameters,
        *,
        exposure_s: float,
        rng: np.random.Generator,
    ) -> TwinObservation:
        expected = self.expected_counts(
            density,
            phase_coordinates,
            parameters,
            exposure_s=exposure_s,
        )
        sampled = self.sample_counts(
            density,
            phase_coordinates,
            parameters,
            exposure_s=exposure_s,
            rng=rng,
        )
        raw, physical, distance = self.reconstruct_density(
            sampled,
            phase_coordinates,
            parameters,
            exposure_s=exposure_s,
        )
        return TwinObservation(
            exposure_s=float(exposure_s),
            expected_counts=expected,
            sampled_counts=sampled,
            reconstructed_rho_raw=raw,
            reconstructed_rho_physical=physical,
            physical_projection_distance=distance,
        )

    def apply_action_and_regenerate(
        self,
        density: np.ndarray,
        phase_coordinates: np.ndarray,
        action_coordinates: np.ndarray,
        parameters: RegisteredTwinParameters,
        *,
        exposure_s: float,
        rng: np.random.Generator,
        action_limit_rad: float = 0.2,
    ) -> Dict[str, Any]:
        coordinates = self._coordinates(phase_coordinates)
        action = self._coordinates(action_coordinates)
        bounded = np.clip(action, -float(action_limit_rad), float(action_limit_rad))
        updated = coordinates + bounded
        observation = self.observe(
            density,
            updated,
            parameters,
            exposure_s=exposure_s,
            rng=rng,
        )
        return {
            "phase_coordinates": updated,
            "bounded_action": bounded,
            "observation": observation,
        }

    def _coordinates(self, values: np.ndarray) -> np.ndarray:
        coordinates = np.asarray(values, dtype=np.float64)
        expected = (self.operation_count, self.coordinate_count)
        if coordinates.shape != expected:
            raise ValueError(f"phase coordinates must have shape {expected}")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("phase coordinates must be finite")
        return coordinates

    @staticmethod
    def _mix_operation_crosstalk(values: np.ndarray, coupling: float) -> np.ndarray:
        return (1.0 - coupling) * values + 0.5 * coupling * (
            np.roll(values, 1) + np.roll(values, -1)
        )

    def _crosstalk_jacobian(
        self,
        local_derivative: np.ndarray,
        coupling: float,
    ) -> np.ndarray:
        jacobian = np.zeros(
            (
                self.operation_count,
                self.operation_count,
                self.coordinate_count,
            ),
            dtype=np.float64,
        )
        for source in range(self.operation_count):
            jacobian[source, source] += (1.0 - coupling) * local_derivative[source]
            jacobian[(source - 1) % self.operation_count, source] += (
                0.5 * coupling * local_derivative[source]
            )
            jacobian[(source + 1) % self.operation_count, source] += (
                0.5 * coupling * local_derivative[source]
            )
        return jacobian

    def _invert_operation_crosstalk(
        self,
        mixed: np.ndarray,
        coupling: float,
    ) -> np.ndarray:
        if coupling <= 0.0:
            return np.asarray(mixed, dtype=np.float64)
        matrix = np.eye(self.operation_count) * (1.0 - coupling)
        for source in range(self.operation_count):
            matrix[(source - 1) % self.operation_count, source] += 0.5 * coupling
            matrix[(source + 1) % self.operation_count, source] += 0.5 * coupling
        return np.linalg.solve(matrix, np.asarray(mixed, dtype=np.float64))
