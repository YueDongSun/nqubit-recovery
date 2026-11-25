"""Immutable synthetic AOD operation schedules with explicit family provenance."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Callable

import numpy as np


SCHEDULE_FAMILIES = frozenset(
    {
        "LEGACY_REAL_3Q_WINDOW_75",
        "LEGACY_REAL_4Q_WINDOW_256",
        "CORE_3Q_TOMOGRAPHY_64",
        "SMALL_N_EXACT_TEACHER",
        "W_COMPLETE_GENERAL_N",
        "PROFILED_DOPT_BUDGETED",
        "BRIDGE_PHASE_REFERENCE",
        "INFORMATION_ONLY_SELECTED",
    }
)


K1_ONE_RDM_COMPLETE_N_SQUARED_ORACLE = "K1_ONE_RDM_COMPLETE_N_SQUARED_ORACLE"


@dataclass(frozen=True)
class Aod_schedule_semantic_scope:
    """Non-serialized scope declaration for a registered schedule family."""

    semantic_alias: str
    supported_excitation_sectors: tuple[int, ...]
    observable_scope: str
    is_full_hilbert_space_povm: bool
    is_generic_fixed_k_measurement: bool
    is_experimental_75_256_parity: bool


K1_ONE_RDM_ORACLE_SCOPE = Aod_schedule_semantic_scope(
    semantic_alias=K1_ONE_RDM_COMPLETE_N_SQUARED_ORACLE,
    supported_excitation_sectors=(1,),
    observable_scope="K1_ONE_RDM_POPULATIONS_AND_PAIR_QUADRATURES",
    is_full_hilbert_space_povm=False,
    is_generic_fixed_k_measurement=False,
    is_experimental_75_256_parity=False,
)


def _immutable_array(value: object, *, dtype: np.dtype, name: str) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind == "b":
        raise ValueError(f"{name} must not use Boolean values")
    result = np.ascontiguousarray(source, dtype=dtype)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return np.frombuffer(result.tobytes(), dtype=result.dtype).reshape(result.shape)


@dataclass(frozen=True)
class Aod_operation_schedule:
    operation_ids: tuple[str, ...]
    amplitudes: np.ndarray
    phases: np.ndarray
    family: str
    development_selected: bool = False
    operation_roles: tuple[str, ...] = ()
    read_throughput: np.ndarray | None = None

    def __post_init__(self) -> None:
        amplitudes = _immutable_array(
            self.amplitudes, dtype=np.dtype(np.float64), name="amplitudes"
        )
        phases = _immutable_array(
            self.phases, dtype=np.dtype(np.float64), name="phases"
        )
        if self.read_throughput is None:
            throughput = np.ones(amplitudes.shape[0], dtype=np.float64)
        else:
            throughput = _immutable_array(
                self.read_throughput, dtype=np.dtype(np.float64), name="read_throughput"
            )
        object.__setattr__(self, "amplitudes", amplitudes)
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "read_throughput", throughput)
        self.validate()

    def validate(self) -> None:
        if type(self.operation_ids) is not tuple or not self.operation_ids:
            raise ValueError("operation_ids must be a nonempty immutable tuple")
        if type(self.operation_roles) is not tuple:
            raise ValueError("operation_roles must be an immutable tuple")
        if self.family not in SCHEDULE_FAMILIES:
            raise ValueError("unregistered AOD schedule family")
        if type(self.development_selected) is not bool:
            raise ValueError("development_selected must be Boolean")
        if (
            self.amplitudes.ndim != 2
            or self.amplitudes.shape != self.phases.shape
            or self.amplitudes.shape[0] != len(self.operation_ids)
        ):
            raise ValueError("operation descriptors must be aligned matrices")
        if self.amplitudes.shape[1] < 3 or self.read_throughput.shape != (
            len(self.operation_ids),
        ):
            raise ValueError("schedule dimensions are invalid")
        if not self.operation_roles:
            raise ValueError(
                "operation_roles are required for canonical schedule identity"
            )
        if len(self.operation_roles) != len(self.operation_ids) or len(
            set(self.operation_ids)
        ) != len(self.operation_ids):
            raise ValueError("operation IDs and roles must align with unique IDs")
        if any(
            type(operation_id) is not str or not operation_id
            for operation_id in self.operation_ids + self.operation_roles
        ):
            raise ValueError("operation IDs and roles cannot be empty")
        if (
            not np.all(np.isfinite(self.amplitudes))
            or not np.all(np.isfinite(self.phases))
            or not np.all(np.isfinite(self.read_throughput))
        ):
            raise ValueError("schedule descriptors must be finite")
        if (
            np.any(self.amplitudes < 0.0)
            or np.any(self.read_throughput < 0.0)
            or np.any(self.read_throughput > 1.0)
        ):
            raise ValueError("amplitudes and read throughput are invalid")
        if np.any(np.count_nonzero(self.amplitudes, axis=1) == 0):
            raise ValueError("every operation must have at least one active mode")

    @property
    def n_mode(self) -> int:
        return int(self.amplitudes.shape[1])

    @property
    def semantic_scope(self) -> Aod_schedule_semantic_scope | None:
        """Return non-serialized measurement scope without changing schedule identity."""
        if self.family == "W_COMPLETE_GENERAL_N":
            return K1_ONE_RDM_ORACLE_SCOPE
        return None

    def sha256(self) -> str:
        self.validate()
        descriptor = {
            "operation_ids": self.operation_ids,
            "operation_roles": self.operation_roles,
            "family": self.family,
            "development_selected": self.development_selected,
            "arrays": [],
        }
        payload = bytearray(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
        )
        for name, array in (
            ("amplitudes", self.amplitudes),
            ("phases", self.phases),
            ("read_throughput", self.read_throughput),
        ):
            payload.extend(
                json.dumps(
                    {"name": name, "dtype": array.dtype.str, "shape": array.shape},
                    separators=(",", ":"),
                ).encode()
            )
            payload.extend(np.ascontiguousarray(array).tobytes())
        return hashlib.sha256(payload).hexdigest()


def population_pair_quadrature_candidates(n_mode: int) -> Aod_operation_schedule:
    """Build the registered synthetic candidate pool, not a real-core claim."""
    if type(n_mode) is not int or n_mode < 3:
        raise ValueError("n_mode must be an integer of at least three")
    rows: list[tuple[np.ndarray, np.ndarray]] = []
    operation_ids: list[str] = []
    roles: list[str] = []
    for mode in range(n_mode):
        amplitude = np.zeros(n_mode)
        amplitude[mode] = 1.0
        rows.append((amplitude, np.zeros(n_mode)))
        operation_ids.append(f"population_{mode}")
        roles.append("population")
    for left in range(n_mode):
        for right in range(left + 1, n_mode):
            amplitude = np.zeros(n_mode)
            amplitude[[left, right]] = 1.0
            for label, phase_value in (("x", 0.0), ("y", np.pi / 2.0)):
                phase = np.zeros(n_mode)
                phase[right] = phase_value
                rows.append((amplitude, phase))
                operation_ids.append(f"pair_{left}_{right}_{label}")
                roles.append("pair_quadrature")
    return Aod_operation_schedule(
        tuple(operation_ids),
        np.asarray([row[0] for row in rows]),
        np.asarray([row[1] for row in rows]),
        "W_COMPLETE_GENERAL_N",
        operation_roles=tuple(roles),
    )


SCHEDULE_BUILDERS: dict[str, Callable[[int], Aod_operation_schedule]] = {
    "W_COMPLETE_GENERAL_N": population_pair_quadrature_candidates
}


def build_registered_synthetic_schedule(
    family: str, n_mode: int
) -> Aod_operation_schedule:
    """Build only registered synthetic families; real legacy families need a source binding."""
    builder = SCHEDULE_BUILDERS.get(family)
    if builder is None:
        raise ValueError(
            "family has no synthetic builder; do not fabricate a legacy real schedule"
        )
    return builder(n_mode)
