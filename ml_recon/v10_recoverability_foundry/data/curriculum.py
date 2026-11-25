"""Registered D0-D7 Track A curriculum cells."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

from ml_recon.physics_hybrid_v10.contracts import sha256_of

from ..config import DENSE_N_HARD_LIMIT
from ..physics.collision import CollisionArm


EXECUTABLE_TRACK_A_STAGES = frozenset({"D0", "D1", "D2"})


@dataclass(frozen=True)
class TrackACurriculumCell:
    stage: str
    n: int
    k: int
    state_family: str
    measurement_family: str
    noise_family: str
    budget_mode: str
    collision_arm: str = ""

    @property
    def cell_id(self) -> str:
        identity = (
            f"{self.stage}:N{self.n}:K{self.k}:{self.state_family}:"
            f"{self.measurement_family}:{self.noise_family}:{self.budget_mode}"
        )
        return f"{identity}:{self.collision_arm}" if self.collision_arm else identity

    @property
    def paired_cell_id(self) -> str:
        """Identity shared by controlled collision arms but never across splits/ranks."""

        return (
            f"{self.stage}:N{self.n}:K{self.k}:{self.state_family}:"
            f"{self.measurement_family}:{self.noise_family}:{self.budget_mode}"
        )

    @property
    def renderer_available(self) -> bool:
        return bool(
            (self.stage == "D0" and self.n in (3, 4) and self.k == 1
             and self.measurement_family == "full_pauli")
            or (self.stage == "D1" and self.n <= DENSE_N_HARD_LIMIT
                and self.measurement_family == "incomplete_pauli")
            or (self.stage == "D2" and self.n <= DENSE_N_HARD_LIMIT
                and self.k == 2 and self.measurement_family == "fock_interference")
        )

    def validate(self) -> None:
        if self.stage not in {f"D{index}" for index in range(8)}:
            raise ValueError("stage must be D0..D7")
        if not 3 <= self.n <= 256 or not 1 <= self.k <= min(5, self.n):
            raise ValueError("curriculum N/K outside the registered range")
        if self.budget_mode not in {"S-total", "S-setting", "S-ideal"}:
            raise ValueError("unknown shot-budget mode")
        if self.stage in EXECUTABLE_TRACK_A_STAGES and not self.renderer_available:
            raise ValueError(
                f"{self.stage} is outside its registered exact renderer boundary")
        if self.stage == "D2":
            if self.noise_family != "collision_three_arm":
                raise ValueError("D2 is reserved for the registered collision comparison")
            CollisionArm(self.collision_arm)
        elif self.collision_arm:
            raise ValueError("collision_arm is legal only for D2 cells")

    def sha256(self) -> str:
        self.validate()
        return sha256_of(asdict(self))


@dataclass(frozen=True)
class TrackACurriculum:
    cells: tuple[TrackACurriculumCell, ...]
    stage_token_quota: Dict[str, int]

    def validate(self) -> None:
        if not self.cells or len({cell.cell_id for cell in self.cells}) != len(self.cells):
            raise ValueError("curriculum cells must be non-empty and unique")
        for cell in self.cells:
            cell.validate()
        stages = {cell.stage for cell in self.cells}
        if stages != {f"D{index}" for index in range(8)}:
            raise ValueError("curriculum must cover every D0-D7 stage")
        if set(self.stage_token_quota) != stages or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in self.stage_token_quota.values()):
            raise ValueError("every stage requires a positive token quota")

    def cells_for_stage(self, stage: str) -> tuple[TrackACurriculumCell, ...]:
        selected = tuple(cell for cell in self.cells if cell.stage == stage)
        if not selected:
            raise ValueError(f"unknown or empty curriculum stage {stage}")
        return selected


def default_curriculum() -> TrackACurriculum:
    cells = (
        TrackACurriculumCell("D0", 4, 1, "exact_fixture", "full_pauli", "clean", "S-ideal"),
        TrackACurriculumCell("D1", 8, 1, "w_sector", "incomplete_pauli", "isolated", "S-total"),
        *tuple(
            TrackACurriculumCell(
                "D2",
                8,
                2,
                "fixed_k",
                "fock_interference",
                "collision_three_arm",
                "S-total",
                arm.value,
            )
            for arm in CollisionArm
        ),
        TrackACurriculumCell(
            "D3", 24, 3, "sparse_fixed_k", "b1_prime", "held_out_backend", "S-total"
        ),
        TrackACurriculumCell(
            "D4", 32, 2, "sparse_fixed_k", "b1_prime", "nuisance_profiled", "S-setting"
        ),
        TrackACurriculumCell("D5", 64, 3, "functional", "decision_aware", "compound", "S-total"),
        TrackACurriculumCell(
            "D6", 128, 4, "functional", "active_boundary", "near_boundary", "S-total"
        ),
        TrackACurriculumCell("D7", 256, 5, "functional", "confirmation_grid", "stress", "S-total"),
    )
    result = TrackACurriculum(
        cells=cells,
        stage_token_quota={f"D{index}": 100_000 for index in range(8)},
    )
    result.validate()
    return result
