"""Executable K=2 hard-core/collision arm semantics and shared measurements."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from ml_recon.physics_hybrid_v10.contracts import sha256_of
from ml_recon.physics_hybrid_v10.shot_budget import (
    S_TOTAL,
    V10ShotBudget,
    allocate_shots,
)
from ml_recon.physics_hybrid_v8.fock_leakage import (
    bosonic_occupation_basis,
    bosonic_sector_transition_matrix,
)

from ..config import DENSE_N_HARD_LIMIT
from ..contracts.core import validate_hex_digest
from .fixed_k import FixedKState


FOCK_BOSON_NUMBER = 2
COLLISION_MEASUREMENT_SCHEMA = "V10TrackACollisionMeasurementV1"
D2_COLLISION_RENDERER_IDENTITY_SCHEMA = "V10D2CollisionRendererIdentityV1"
D2_COLLISION_RENDERER_IDENTITY_KIND = "SYNTHETIC_FOCK_INTERFERENCE_HYPOTHESIS_UNBOUND"
D2_COLLISION_RENDERER_IDENTITY_STATUS = "HYPOTHESIS_UNBOUND"
_D2_EXECUTION_DEPENDENCY_SCHEMA = "V10D2BoundedExecutionClosureV1"
_D2_EXECUTION_DEPENDENCY_PATHS = (
    "ml_recon/v10_recoverability_foundry/physics/collision.py",
    "ml_recon/v10_recoverability_foundry/data/measurement.py",
    "ml_recon/v10_recoverability_foundry/physics/fixed_k.py",
    "ml_recon/v10_recoverability_foundry/data/curriculum.py",
    "ml_recon/v10_recoverability_foundry/config.py",
    "ml_recon/v10_recoverability_foundry/contracts/core.py",
    "ml_recon/physics_hybrid_v8/fock_leakage.py",
    "ml_recon/physics_hybrid_v10/contracts.py",
    "ml_recon/physics_hybrid_v10/shot_budget.py",
    "ml_recon/physics_hybrid_v10/measurement_model.py",
    "ml_recon/physics_hybrid_v10/measurement_scheduler.py",
    "ml_recon/physics_hybrid_v10/state_backends.py",
    "ml_recon/physics_hybrid_v10/synthetic_manifest.py",
)


class CollisionArm(str, Enum):
    COHERENT_EXACT = "coherent_exact"
    SECTOR_DEPHASED = "sector_dephased"
    CLASSICAL_MIXTURE = "classical_mixture"


def _source_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _d2_renderer_source_hashes() -> dict[str, str]:
    repository_root = Path(__file__).resolve().parents[3]
    return {
        relative_path: _source_file_sha256(repository_root / relative_path)
        for relative_path in _D2_EXECUTION_DEPENDENCY_PATHS
    }


@dataclass(frozen=True)
class D2CollisionRendererIdentity:
    """Synthetic, unbound identity for the public exact D2 renderer.

    This is deliberately not a hardware receipt, calibration record, or
    authority grant.  It binds the three-arm law to its public source and
    settings so research adapters cannot substitute arbitrary digest strings.
    """

    cell_sha: str
    collision_arm: CollisionArm
    collision_measurement_contract: "CollisionMeasurementContract"
    shot_allocation: V10ShotBudget
    collision_measurement_contract_sha: str
    noise_contract_sha: str
    apparatus_contract_sha: str
    control_set_contract_sha: str
    measurement_renderer_source_sha: str
    collision_physics_source_sha: str
    execution_dependency_manifest: tuple[tuple[str, str], ...]
    execution_dependency_manifest_sha: str
    renderer_source_sha: str
    identity_kind: str = D2_COLLISION_RENDERER_IDENTITY_KIND
    identity_status: str = D2_COLLISION_RENDERER_IDENTITY_STATUS
    schema: str = D2_COLLISION_RENDERER_IDENTITY_SCHEMA

    @classmethod
    def build(
        cls,
        *,
        cell_sha: str,
        collision_arm: CollisionArm | str,
        collision_measurement_contract: "CollisionMeasurementContract",
        shot_allocation: V10ShotBudget,
    ) -> "D2CollisionRendererIdentity":
        validate_hex_digest(cell_sha, field_name="cell_sha")
        arm = CollisionArm(collision_arm)
        collision_measurement_contract.validate()
        shot_allocation.validate()
        canonical_contract = default_collision_measurement_contract(
            collision_measurement_contract.mode_count
        )
        if (
            collision_measurement_contract.sha256() != canonical_contract.sha256()
            or collision_measurement_contract.setting_ids
            != canonical_contract.setting_ids
            or len(collision_measurement_contract.unitaries)
            != len(canonical_contract.unitaries)
            or any(
                not np.array_equal(actual, expected)
                for actual, expected in zip(
                    collision_measurement_contract.unitaries,
                    canonical_contract.unitaries,
                )
            )
        ):
            raise ValueError(
                "D2 collision identity requires the canonical public exact measurement law"
            )
        if shot_allocation.mode != S_TOTAL:
            raise ValueError("D2 collision identity requires S-total allocation")
        if (
            shot_allocation.setting_ids != collision_measurement_contract.setting_ids
            or set(shot_allocation.shots_by_setting)
            != set(collision_measurement_contract.setting_ids)
        ):
            raise ValueError("D2 allocation must bind the collision setting registry")
        canonical_allocation = allocate_shots(
            canonical_contract.setting_ids,
            S_TOTAL,
            shot_allocation.total_shots,
        )
        if shot_allocation.sha256() != canonical_allocation.sha256():
            raise ValueError(
                "D2 collision identity allocation must equal canonical S-total allocation"
            )
        source_hashes = _d2_renderer_source_hashes()
        measurement_source_sha = source_hashes[
            "ml_recon/v10_recoverability_foundry/data/measurement.py"
        ]
        collision_source_sha = source_hashes[
            "ml_recon/v10_recoverability_foundry/physics/collision.py"
        ]
        measurement_sha = collision_measurement_contract.sha256()
        noise_sha = sha256_of(
            {
                "schema": "V10D2CollisionThreeArmNoiseIdentityV1",
                "identity_kind": D2_COLLISION_RENDERER_IDENTITY_KIND,
                "identity_status": D2_COLLISION_RENDERER_IDENTITY_STATUS,
                "cell_sha": cell_sha,
                "collision_arm": arm.value,
                "collision_measurement_contract_sha": measurement_sha,
                "collision_physics_source_sha": collision_source_sha,
                "measurement_renderer_source_sha": measurement_source_sha,
            }
        )
        apparatus_sha = sha256_of(
            {
                "schema": "V10D2SyntheticFockInterferenceApparatusV1",
                "identity_kind": D2_COLLISION_RENDERER_IDENTITY_KIND,
                "identity_status": D2_COLLISION_RENDERER_IDENTITY_STATUS,
                "mode_count": collision_measurement_contract.mode_count,
                "boson_number": FOCK_BOSON_NUMBER,
                "number_resolving": True,
                "collision_measurement_contract_sha": measurement_sha,
                "collision_physics_source_sha": collision_source_sha,
                "measurement_renderer_source_sha": measurement_source_sha,
            }
        )
        control_sha = sha256_of(
            {
                "schema": "V10D2FockInterferenceControlSetV1",
                "identity_kind": D2_COLLISION_RENDERER_IDENTITY_KIND,
                "identity_status": D2_COLLISION_RENDERER_IDENTITY_STATUS,
                "cell_sha": cell_sha,
                "collision_measurement_contract_sha": measurement_sha,
                "setting_ids": list(collision_measurement_contract.setting_ids),
                "measurement_renderer_source_sha": measurement_source_sha,
            }
        )
        renderer_source_sha = sha256_of(
            {
                "schema": "V10D2RendererSourceIdentityV1",
                "bounded_execution_dependency_manifest": source_hashes,
            }
        )
        dependency_manifest_sha = sha256_of(
            {
                "schema": _D2_EXECUTION_DEPENDENCY_SCHEMA,
                "dependencies": source_hashes,
            }
        )
        return cls(
            cell_sha=cell_sha,
            collision_arm=arm,
            collision_measurement_contract=collision_measurement_contract,
            shot_allocation=shot_allocation,
            collision_measurement_contract_sha=measurement_sha,
            noise_contract_sha=noise_sha,
            apparatus_contract_sha=apparatus_sha,
            control_set_contract_sha=control_sha,
            measurement_renderer_source_sha=measurement_source_sha,
            collision_physics_source_sha=collision_source_sha,
            execution_dependency_manifest=tuple(sorted(source_hashes.items())),
            execution_dependency_manifest_sha=dependency_manifest_sha,
            renderer_source_sha=renderer_source_sha,
        )

    def validate(self) -> None:
        if self.schema != D2_COLLISION_RENDERER_IDENTITY_SCHEMA:
            raise ValueError("unknown D2 collision renderer identity schema")
        if self.identity_kind != D2_COLLISION_RENDERER_IDENTITY_KIND:
            raise ValueError(
                "D2 collision identity must remain synthetic and hypothesis-unbound"
            )
        if self.identity_status != D2_COLLISION_RENDERER_IDENTITY_STATUS:
            raise ValueError("D2 collision identity must remain HYPOTHESIS_UNBOUND")
        expected = self.build(
            cell_sha=self.cell_sha,
            collision_arm=self.collision_arm,
            collision_measurement_contract=self.collision_measurement_contract,
            shot_allocation=self.shot_allocation,
        )
        for name in (
            "collision_measurement_contract_sha",
            "noise_contract_sha",
            "apparatus_contract_sha",
            "control_set_contract_sha",
            "measurement_renderer_source_sha",
            "collision_physics_source_sha",
            "execution_dependency_manifest",
            "execution_dependency_manifest_sha",
            "renderer_source_sha",
        ):
            if getattr(self, name) != getattr(expected, name):
                raise ValueError(
                    f"D2 collision identity {name} does not match public renderer"
                )

    def sha256(self) -> str:
        self.validate()
        return sha256_of(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the scalar identity projection, never a renderer object graph."""

        return {
            "schema": self.schema,
            "cell_sha": self.cell_sha,
            "collision_arm": self.collision_arm.value,
            "collision_measurement_contract_sha": self.collision_measurement_contract_sha,
            "canonical_shot_allocation_sha": self.shot_allocation.sha256(),
            "noise_contract_sha": self.noise_contract_sha,
            "apparatus_contract_sha": self.apparatus_contract_sha,
            "control_set_contract_sha": self.control_set_contract_sha,
            "measurement_renderer_source_sha": self.measurement_renderer_source_sha,
            "collision_physics_source_sha": self.collision_physics_source_sha,
            "execution_dependency_manifest": dict(self.execution_dependency_manifest),
            "execution_dependency_manifest_sha": self.execution_dependency_manifest_sha,
            "renderer_source_sha": self.renderer_source_sha,
            "identity_kind": self.identity_kind,
            "identity_status": self.identity_status,
        }


@dataclass(frozen=True)
class CollisionMeasurementContract:
    """One deterministic exact Fock-sector schedule shared by all three arms."""

    setting_ids: tuple[str, ...]
    unitaries: tuple[np.ndarray, ...]
    schema: str = COLLISION_MEASUREMENT_SCHEMA

    @property
    def mode_count(self) -> int:
        return int(np.asarray(self.unitaries[0]).shape[0])

    def validate(self) -> None:
        if self.schema != COLLISION_MEASUREMENT_SCHEMA:
            raise ValueError("unknown collision measurement schema")
        if (
            not self.setting_ids
            or len(self.setting_ids) != len(self.unitaries)
            or len(set(self.setting_ids)) != len(self.setting_ids)
        ):
            raise ValueError(
                "collision settings must be non-empty, unique, and aligned"
            )
        mode_count = int(np.asarray(self.unitaries[0]).shape[0])
        if not 3 <= mode_count <= DENSE_N_HARD_LIMIT:
            raise ValueError("exact collision measurements require 3<=N<=16")
        identity = np.eye(mode_count, dtype=np.complex128)
        for unitary in self.unitaries:
            value = np.asarray(unitary, dtype=np.complex128)
            if (
                value.shape != (mode_count, mode_count)
                or not np.all(np.isfinite(value))
                or not np.allclose(value.conj().T @ value, identity, atol=1e-9)
            ):
                raise ValueError(
                    "collision measurement transforms must be finite unitaries"
                )

    def sha256(self) -> str:
        self.validate()
        return sha256_of(
            {
                "schema": self.schema,
                "setting_ids": list(self.setting_ids),
                "unitaries": [
                    {"real": value.real.tolist(), "imag": value.imag.tolist()}
                    for value in (
                        np.asarray(unitary, dtype=np.complex128)
                        for unitary in self.unitaries
                    )
                ],
            }
        )


def default_collision_measurement_contract(
    mode_count: int,
) -> CollisionMeasurementContract:
    """Build identity and two phase-sensitive exact two-boson analyzers."""

    n = int(mode_count)
    if not 3 <= n <= DENSE_N_HARD_LIMIT:
        raise ValueError("collision contract is exact only for 3<=N<=16")
    index = np.arange(n, dtype=np.float64)
    fourier = np.exp(2j * np.pi * np.outer(index, index) / n) / np.sqrt(n)
    phase = np.exp(1j * np.pi * np.square(index) / n)
    phased_fourier = fourier @ np.diag(phase)
    result = CollisionMeasurementContract(
        setting_ids=("fock_identity", "fock_fourier", "fock_phased_fourier"),
        unitaries=(np.eye(n, dtype=np.complex128), fourier, phased_fourier),
    )
    result.validate()
    return result


@dataclass(frozen=True)
class CollisionArmState:
    hard_core_state_sha: str
    arm: CollisionArm
    hard_core_mass: float
    collision_mass: float
    cross_sector_coherence_norm: float
    approximation: bool
    occupation_basis: np.ndarray
    sector_density: np.ndarray

    def validate(self) -> None:
        validate_hex_digest(self.hard_core_state_sha, field_name="hard_core_state_sha")
        basis = np.asarray(self.occupation_basis, dtype=np.int64)
        density = np.asarray(self.sector_density, dtype=np.complex128)
        if (
            basis.ndim != 2
            or basis.shape[1] < 3
            or not np.all(np.sum(basis, axis=1) == FOCK_BOSON_NUMBER)
        ):
            raise ValueError(
                "collision occupation basis must be an exact Fock-2 registry"
            )
        if density.shape != (basis.shape[0], basis.shape[0]):
            raise ValueError(
                "collision density and occupation basis dimensions disagree"
            )
        if (
            not np.all(np.isfinite(density))
            or not np.allclose(density, density.conj().T, atol=1e-10)
            or float(np.linalg.eigvalsh(density).min(initial=0.0)) < -1e-10
            or not np.isclose(float(np.trace(density).real), 1.0, atol=1e-10)
        ):
            raise ValueError(
                "collision sector density must be finite, Hermitian, PSD, and trace one"
            )
        if not np.isclose(self.hard_core_mass + self.collision_mass, 1.0, atol=1e-12):
            raise ValueError("hard-core and collision masses must sum to one")
        if (
            min(
                self.hard_core_mass,
                self.collision_mass,
                self.cross_sector_coherence_norm,
            )
            < 0.0
        ):
            raise ValueError("collision-arm masses and coherence must be non-negative")
        if self.arm is not CollisionArm.COHERENT_EXACT and not np.isclose(
            self.cross_sector_coherence_norm, 0.0, atol=1e-14
        ):
            raise ValueError(
                "dephased/mixture arms cannot retain cross-sector coherence"
            )
        if self.approximation != (self.arm is CollisionArm.CLASSICAL_MIXTURE):
            raise ValueError("only the classical-mixture arm may be marked approximate")

    def measurement_probabilities(self, unitary: np.ndarray) -> np.ndarray:
        """Evaluate the exact number-resolving Fock-2 law for one shared analyzer."""

        self.validate()
        transition, input_basis, output_basis = bosonic_sector_transition_matrix(
            np.asarray(unitary, dtype=np.complex128),
            boson_number=FOCK_BOSON_NUMBER,
        )
        if not np.array_equal(input_basis, self.occupation_basis) or not np.array_equal(
            output_basis, self.occupation_basis
        ):
            raise RuntimeError("collision measurement basis registry changed")
        output_density = transition @ self.sector_density @ transition.conj().T
        probabilities = np.clip(np.real(np.diag(output_density)), 0.0, None)
        total = float(probabilities.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise FloatingPointError("collision measurement law is not normalizable")
        probabilities /= total
        return probabilities


def _principal_collision_mode(state: FixedKState) -> np.ndarray:
    values, vectors = np.linalg.eigh(state.one_rdm())
    mode = np.asarray(vectors[:, int(np.argmax(values))], dtype=np.complex128)
    pivot = int(np.argmax(np.abs(mode)))
    mode *= np.exp(-1j * np.angle(mode[pivot]))
    return mode / max(float(np.linalg.norm(mode)), 1e-15)


def build_collision_arm(
    state: FixedKState,
    *,
    collision_mass: float,
    arm: CollisionArm | str,
    collision_mode: np.ndarray | None = None,
) -> CollisionArmState:
    """Embed one paired latent into coherent, dephased, or classical Fock-2 density."""

    if state.k != FOCK_BOSON_NUMBER:
        raise ValueError("same-mode Fock-2 collision arms require K=2")
    if state.n > DENSE_N_HARD_LIMIT:
        raise ValueError("exact collision arms are forbidden above N=16")
    selected = CollisionArm(arm)
    if not 0.0 <= collision_mass <= 1.0:
        raise ValueError("collision_mass must lie in [0,1]")
    hard_core_mass = 1.0 - float(collision_mass)
    basis = bosonic_occupation_basis(state.n, FOCK_BOSON_NUMBER)
    basis_index = {tuple(row.tolist()): index for index, row in enumerate(basis)}

    hard_core_amplitude = np.zeros(basis.shape[0], dtype=np.complex128)
    for modes, amplitude in zip(state.support, state.amplitudes):
        occupation = np.zeros(state.n, dtype=np.int64)
        occupation[list(modes)] = 1
        hard_core_amplitude[basis_index[tuple(occupation.tolist())]] = amplitude

    mode = (
        _principal_collision_mode(state)
        if collision_mode is None
        else np.asarray(collision_mode, dtype=np.complex128).reshape(-1)
    )
    if mode.shape != (state.n,) or not np.all(np.isfinite(mode)):
        raise ValueError("collision_mode must be one finite mode vector")
    mode = mode / max(float(np.linalg.norm(mode)), 1e-15)
    collision_amplitude = np.zeros(basis.shape[0], dtype=np.complex128)
    for mode_index in range(state.n):
        occupation = np.zeros(state.n, dtype=np.int64)
        occupation[mode_index] = FOCK_BOSON_NUMBER
        collision_amplitude[basis_index[tuple(occupation.tolist())]] = (
            mode[mode_index] ** 2
        )
    collision_amplitude /= max(float(np.linalg.norm(collision_amplitude)), 1e-15)

    hard_density = np.outer(hard_core_amplitude, hard_core_amplitude.conj())
    collision_density = np.outer(collision_amplitude, collision_amplitude.conj())
    if selected is CollisionArm.COHERENT_EXACT:
        amplitude = (
            np.sqrt(hard_core_mass) * hard_core_amplitude
            + np.sqrt(float(collision_mass)) * collision_amplitude
        )
        density = np.outer(amplitude, amplitude.conj())
        coherence = float(np.sqrt(hard_core_mass * float(collision_mass)))
    else:
        density = (
            hard_core_mass * hard_density + float(collision_mass) * collision_density
        )
        coherence = 0.0
        if selected is CollisionArm.CLASSICAL_MIXTURE:
            density = np.diag(np.real(np.diag(density))).astype(np.complex128)

    result = CollisionArmState(
        hard_core_state_sha=state.state_sha(),
        arm=selected,
        hard_core_mass=hard_core_mass,
        collision_mass=float(collision_mass),
        cross_sector_coherence_norm=coherence,
        approximation=selected is CollisionArm.CLASSICAL_MIXTURE,
        occupation_basis=basis,
        sector_density=density,
    )
    result.validate()
    return result
