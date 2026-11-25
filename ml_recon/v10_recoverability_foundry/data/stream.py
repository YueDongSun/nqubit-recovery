"""Rank-disjoint, split-explicit, quota-bounded Track A sample stream."""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterator

import numpy as np

from ml_recon.physics_hybrid_v10.contracts import sha256_of
from ml_recon.physics_hybrid_v10.synthetic_manifest import V10CounterBasedRng

from ..config import TrackAConfig
from ..contracts import TrackASample, TrackASampleKey
from ..physics.fixed_k import FixedKState, random_fixed_k_state
from .curriculum import TrackACurriculum, TrackACurriculumCell, default_curriculum
from .measurement import planned_physical_token_count, render_track_a_measurement


class TrackAStream:
    def __init__(
        self,
        config: TrackAConfig,
        *,
        stage: str,
        split: str = "train",
        rank: int = 0,
        world_size: int = 1,
        curriculum: TrackACurriculum | None = None,
    ):
        config.validate()
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank must lie in [0,world_size)")
        self.config = config
        self.stage = stage
        self.split = split
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.curriculum = curriculum or default_curriculum()
        self.curriculum.validate()
        self.cells = self.curriculum.cells_for_stage(stage)

    def key(
        self,
        *,
        global_step: int,
        microbatch: int,
        slot: int,
        cell_index: int | None = None,
    ) -> TrackASampleKey:
        selected_index = int(global_step) if cell_index is None else int(cell_index)
        cell = self.cells[selected_index % len(self.cells)]
        key = TrackASampleKey(
            split=self.split,
            stage=self.stage,
            cell_id=cell.cell_id,
            global_step=int(global_step),
            rank=self.rank,
            microbatch=int(microbatch),
            slot=int(slot),
            global_seed=self.config.data.global_seed,
        )
        key.validate()
        return key

    @staticmethod
    def _paired_sample_id(key: TrackASampleKey, cell: TrackACurriculumCell) -> str:
        return (
            f"track_a_pair:{key.split}:{key.stage}:{cell.paired_cell_id}:"
            f"step{key.global_step}:rank{key.rank}:micro{key.microbatch}:"
            f"slot{key.slot}:seed{key.global_seed}"
        )

    @staticmethod
    def _state_for_cell(
        cell: TrackACurriculumCell,
        rng: V10CounterBasedRng,
        max_sparse_terms: int,
    ) -> FixedKState:
        if cell.stage == "D0":
            support = tuple((index,) for index in range(cell.n))
            return FixedKState(
                n=cell.n,
                k=cell.k,
                support=support,
                amplitudes=np.ones(cell.n, dtype=np.complex128),
            )
        return random_fixed_k_state(
            cell.n,
            cell.k,
            rng.stream("fixed_k_state"),
            max_sparse_terms=max_sparse_terms,
        )

    def render(self, key: TrackASampleKey) -> TrackASample:
        key.validate()
        if key.stage != self.stage or key.split != self.split:
            raise ValueError("sample key stage/split does not match this stream")
        cell = next((value for value in self.cells if value.cell_id == key.cell_id), None)
        if cell is None:
            raise ValueError("sample key references a cell outside this stream")
        if not cell.renderer_available:
            raise NotImplementedError(
                f"{cell.stage} renderer is unimplemented; labels cannot replace observations")

        paired_sample_id = self._paired_sample_id(key, cell)
        sample_rng = V10CounterBasedRng(
            key.sample_id, global_seed=key.global_seed, split=key.split)
        paired_rng = V10CounterBasedRng(
            paired_sample_id, global_seed=key.global_seed, split=key.split)
        state = self._state_for_cell(
            cell,
            paired_rng if cell.stage == "D2" else sample_rng,
            self.config.physics.max_sparse_terms,
        )
        functionals = state.selected_functionals()
        measurement = render_track_a_measurement(
            cell=cell,
            state=state,
            sample_id=key.sample_id,
            budget=self.config.data.smoke_shots,
            rng=sample_rng,
            paired_rng=paired_rng,
        )
        planned_tokens = planned_physical_token_count(
            cell, self.config.data.smoke_shots)
        if measurement.physical_token_count != planned_tokens:
            raise RuntimeError("rendered physical-token count differs from the registered plan")
        if planned_tokens > self.config.data.token_budget_per_batch:
            raise RuntimeError("sample exceeds token_budget_per_batch")

        sample = TrackASample(
            key=key,
            n=cell.n,
            k=cell.k,
            observation=measurement.observation,
            truth={
                "fixed_k_state": state,
                "selected_functionals": functionals,
                "state_sha": state.state_sha(),
                **measurement.truth,
            },
            metadata={
                "sample_id": key.sample_id,
                "key_sha": key.sha256(),
                "split": key.split,
                "cell": asdict(cell),
                "paired_sample_id": paired_sample_id,
                "paired_latent_sha": sha256_of(paired_sample_id),
                "paired_crn_sha": measurement.paired_crn_sha,
                "measurement_contract_sha": measurement.measurement_contract_sha,
                "physical_token_count": planned_tokens,
                "ephemeral": True,
                "research_only": True,
            },
        )
        sample.validate()
        return sample

    def iter_steps(self, start_step: int, stop_step: int) -> Iterator[TrackASample]:
        if not 0 <= start_step <= stop_step:
            raise ValueError("invalid step interval")
        costs = tuple(
            planned_physical_token_count(cell, self.config.data.smoke_shots)
            for cell in self.cells
        )
        cycles, remainder = divmod(start_step, len(costs))
        consumed = cycles * sum(costs) + sum(costs[:remainder])
        quota = self.curriculum.stage_token_quota[self.stage]
        for global_step in range(start_step, stop_step):
            cost = costs[global_step % len(costs)]
            if consumed + cost > quota:
                raise RuntimeError(f"stage {self.stage} physical-token quota exhausted")
            sample = self.render(self.key(
                global_step=global_step,
                microbatch=0,
                slot=0,
                cell_index=global_step,
            ))
            consumed += cost
            yield sample
