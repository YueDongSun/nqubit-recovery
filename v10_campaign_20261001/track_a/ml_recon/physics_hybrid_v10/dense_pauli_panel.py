"""Owned immutable small-n_qubits Pauli projector panels for dense V10 backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from .state_backends import (
    DenseDensityStateBackend,
    V10StateBackend,
    all_pauli_outcomes,
    outcome_projector,
)


max_dense_pauli_panel_qubits = 4
max_dense_pauli_panel_settings = 81


def _validated_panel_n(n_qubits: int) -> int:
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, (int, np.integer)):
        raise ValueError("dense Pauli panel n_qubits must be an integer")
    if not 1 <= int(n_qubits) <= max_dense_pauli_panel_qubits:
        raise ValueError(
            "dense Pauli panels support only "
            f"1 <= n_qubits <= {max_dense_pauli_panel_qubits}")
    return int(n_qubits)


def _canonical_axes(n_qubits: int, axes: Sequence[str]) -> Tuple[str, ...]:
    try:
        values = tuple(axes)
    except TypeError as error:
        raise ValueError("Pauli setting must be a sequence") from error
    if len(values) != n_qubits:
        raise ValueError("global Pauli setting length does not match panel n_qubits")
    if any(axis not in ("X", "Y", "Z") for axis in values):
        raise ValueError("physical Pauli axes must be X, Y, or Z")
    return values


def _canonical_outcome(n_qubits: int, outcome: Sequence[int]) -> Tuple[int, ...]:
    try:
        values = tuple(outcome)
    except TypeError as error:
        raise ValueError("Pauli outcome must be a sequence") from error
    if len(values) != n_qubits:
        raise ValueError("global Pauli outcome length does not match panel n_qubits")
    try:
        canonical = tuple(int(value) for value in values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Pauli outcomes must be +/-1") from error
    if any(value not in (-1, 1) for value in canonical):
        raise ValueError("Pauli outcomes must be +/-1")
    return canonical


def _canonical_settings(
    n_qubits: int,
    settings: Sequence[Sequence[str]],
) -> Tuple[Tuple[str, ...], ...]:
    try:
        values = tuple(settings)
    except TypeError as error:
        raise ValueError("Pauli settings must be a sequence") from error
    if not values:
        raise ValueError("dense Pauli panel requires at least one setting")
    return tuple(_canonical_axes(n_qubits, axes) for axes in values)


def _legacy_projector(
    axes: Tuple[str, ...],
    outcome: Tuple[int, ...],
) -> np.ndarray:
    """Construct a projector in the exact legacy DenseDensity backend order."""
    projector = np.array([[1.0 + 0.0j]])
    for axis, value in zip(axes, outcome):
        projector = np.kron(projector, outcome_projector(axis, int(value)))
    return projector


@dataclass(frozen=True)
class Dense_Pauli_Panel:
    """Immutable full-Pauli projectors owned by one explicit render operation."""

    N: int
    settings: Tuple[Tuple[str, ...], ...]
    _outcomes_payload: bytes
    _projector_payloads: Mapping[
        Tuple[Tuple[str, ...], Tuple[int, ...]],
        bytes,
    ]
    _settings_set: frozenset[Tuple[str, ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        n_qubits = _validated_panel_n(self.N)
        settings = tuple(dict.fromkeys(_canonical_settings(n_qubits, self.settings)))
        if len(settings) > min(3 ** n_qubits, max_dense_pauli_panel_settings):
            raise ValueError("dense Pauli panel exceeds its setting bound")

        outcomes = all_pauli_outcomes(n_qubits)
        try:
            outcomes_payload = bytes(self._outcomes_payload)
        except (TypeError, ValueError) as error:
            raise ValueError("compiled panel outcomes must use immutable bytes") from error
        if outcomes_payload != outcomes.tobytes():
            raise ValueError("compiled panel outcomes do not match legacy ordering")

        expected_keys = {
            (axes, tuple(int(value) for value in outcome))
            for axes in settings
            for outcome in outcomes
        }
        expected_payload_size = (2 ** n_qubits) ** 2 * np.dtype(np.complex128).itemsize
        try:
            payload_items = self._projector_payloads.items()
        except AttributeError as error:
            raise ValueError("compiled panel projector payloads must be a mapping") from error

        payloads: Dict[Tuple[Tuple[str, ...], Tuple[int, ...]], bytes] = {}
        for raw_key, raw_payload in payload_items:
            try:
                raw_axes, raw_outcome = raw_key
            except (TypeError, ValueError) as error:
                raise ValueError("compiled panel projector key is invalid") from error
            key = (
                _canonical_axes(n_qubits, raw_axes),
                _canonical_outcome(n_qubits, raw_outcome),
            )
            try:
                payload = bytes(raw_payload)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "compiled panel projectors must use immutable bytes") from error
            if len(payload) != expected_payload_size:
                raise ValueError("compiled panel projector payload has invalid size")
            if payload != _legacy_projector(key[0], key[1]).tobytes():
                raise ValueError("compiled panel projector does not match its Pauli axes")
            payloads[key] = payload

        if set(payloads) != expected_keys:
            raise ValueError("compiled panel does not cover its declared settings")

        object.__setattr__(self, "N", n_qubits)
        object.__setattr__(self, "settings", settings)
        object.__setattr__(self, "_outcomes_payload", outcomes_payload)
        object.__setattr__(
            self,
            "_projector_payloads",
            MappingProxyType(payloads),
        )
        object.__setattr__(self, "_settings_set", frozenset(settings))

    @property
    def outcomes(self) -> np.ndarray:
        """Return the immutable lexicographic +/-1 outcome panel."""
        return np.frombuffer(
            self._outcomes_payload,
            dtype=np.int8,
        ).reshape(2 ** self.N, self.N)

    def validate_settings(self, settings: Sequence[Sequence[str]]) -> None:
        for axes in _canonical_settings(self.N, settings):
            if axes not in self._settings_set:
                raise ValueError(
                    "dense Pauli panel does not contain the requested setting")

    def projector(
        self,
        axes: Sequence[str],
        outcome: Sequence[int],
    ) -> np.ndarray:
        axes_key = _canonical_axes(self.N, axes)
        outcome_key = _canonical_outcome(self.N, outcome)
        try:
            payload = self._projector_payloads[(axes_key, outcome_key)]
        except KeyError as error:
            if axes_key not in self._settings_set:
                raise ValueError(
                    "dense Pauli panel does not contain the requested setting") from error
            raise ValueError(
                "dense Pauli panel does not contain the requested outcome") from error
        dimension = 2 ** self.N
        return np.frombuffer(
            payload,
            dtype=np.complex128,
        ).reshape(dimension, dimension)


def compile_dense_pauli_panel(
    n_qubits: int,
    settings: Sequence[Sequence[str]],
) -> Dense_Pauli_Panel:
    """Compile one bounded, immutable panel without using a module cache."""
    n_qubits = _validated_panel_n(n_qubits)
    canonical_settings = tuple(
        dict.fromkeys(_canonical_settings(n_qubits, settings)))
    if len(canonical_settings) > min(3 ** n_qubits, max_dense_pauli_panel_settings):
        raise ValueError("dense Pauli panel exceeds its setting bound")

    outcomes = all_pauli_outcomes(n_qubits)
    payloads: Dict[Tuple[Tuple[str, ...], Tuple[int, ...]], bytes] = {}
    for axes in canonical_settings:
        for outcome in outcomes:
            outcome_key = tuple(int(value) for value in outcome)
            payloads[(axes, outcome_key)] = _legacy_projector(
                axes,
                outcome_key,
            ).tobytes()

    return Dense_Pauli_Panel(
        N=n_qubits,
        settings=canonical_settings,
        _outcomes_payload=outcomes.tobytes(),
        _projector_payloads=payloads,
    )


@dataclass(frozen=True)
class Dense_Pauli_Panel_Backend:
    """Dense backend wrapper that replaces only projector construction."""

    dense_backend: DenseDensityStateBackend
    panel: Dense_Pauli_Panel

    def __post_init__(self) -> None:
        if not isinstance(self.dense_backend, DenseDensityStateBackend):
            raise TypeError("panel backend requires DenseDensityStateBackend")
        if not isinstance(self.panel, Dense_Pauli_Panel):
            raise TypeError("panel must be a Dense_Pauli_Panel")
        if self.dense_backend.N != self.panel.N:
            raise ValueError("dense backend and Pauli panel n_qubits mismatch")

    @property
    def N(self) -> int:
        return self.dense_backend.N

    @property
    def backend_name(self) -> str:
        return self.dense_backend.backend_name

    def probability(self, axes: Sequence[str], outcome: Sequence[int]) -> float:
        projector = self.panel.projector(axes, outcome)
        probability = float(np.trace(self.dense_backend.rho @ projector).real)
        return float(np.clip(probability, 0.0, 1.0))

    def state_sha(self) -> str:
        return self.dense_backend.state_sha()


def bind_dense_pauli_panel(
    backend: V10StateBackend,
    panel: Dense_Pauli_Panel | None,
) -> V10StateBackend:
    """Use the panel for validated dense backends and preserve generic fallback."""
    if panel is None:
        return backend
    if not isinstance(panel, Dense_Pauli_Panel):
        raise TypeError("panel must be a Dense_Pauli_Panel")
    if backend.N != panel.N:
        raise ValueError("backend and Pauli panel n_qubits mismatch")
    if isinstance(backend, DenseDensityStateBackend):
        return Dense_Pauli_Panel_Backend(backend, panel)
    return backend
